from typing import Optional, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import LSTM, BatchNorm1d, Linear, Parameter
from .filtering import wiener
from .transforms import make_filterbanks, ComplexNorm
import typing as tp

from .bandsequence import BandSequenceModelModule
from .bandsplit import BandSplitModule  
from .maskestimation import MaskEstimationModule


def validate_model(model, track, segment_length=255):
    """
    Process a full validation track by segmenting it into smaller parts.
    
    Args:
        model: The trained model.
        track: The input track with shape [1, 2, 2049, T].
        segment_length: The length of the time dimension for each segment.
    
    Returns:
        The reconstructed track with the same shape as the input.
    """
    _, channels, freq_bins, time_steps = track.shape
    output_track = torch.zeros_like(track)  # Placeholder for the reconstructed output

    # Pad the track if necessary
    pad_length = (segment_length - (time_steps % segment_length)) % segment_length
    if pad_length > 0:
        track = F.pad(track, (0, pad_length))  # Pad time dimension
        time_steps += pad_length

    # Process each segment
    for start in range(0, time_steps, segment_length):
        end = start + segment_length
        segment = track[:, :, :, start:end]  # Extract segment
        output_segment = model(segment)  # Run through the model
        output_track[:, :, :, start:end] = output_segment

    # Remove padding from the output if necessary
    if pad_length > 0:
        output_track = output_track[:, :, :, :-pad_length]

    return output_track



class OpenUnmix(nn.Module):
    def __init__(
        self,
        sr: int = 44100,
        n_fft: int = 4096,
        bandsplits: tp.List[tp.Tuple[int, int]]= [
            (1000, 100),
            (4000, 250),
            (8000, 500),
            (16000, 1000),
            (20000, 2000),
        ],
        t_timesteps: int = 255,
        fc_dim: int = 128,
        subband_hidden_size: int = 256,
        subband_lstm_layers: int = 1,
        band_hidden_size: int = 256,
        band_lstm_layers: int = 6,
        mlp_dim: int = 512,
        complex_as_channel: bool = False,
        is_mono: bool = False,
        bidirectional: bool = True,
        rnn_type: str = "LSTM",
        unidirectional: bool = False,
        input_mean: Optional[np.ndarray] = None,
        input_scale: Optional[np.ndarray] = None,
        nb_bins: int = 4096,
        nb_channels: int = 2,
        hidden_size: int = 512,
        nb_layers: int = 3,
        max_bin: Optional[int] = None,
    ):
        super(OpenUnmix, self).__init__()
        self.n_fft= n_fft
        # Band-splitting module
        self.band_split = BandSplitModule(
            sr, n_fft, bandsplits, t_timesteps, fc_dim, complex_as_channel, is_mono
        )
        
        # Band-sequence RNN module
        self.band_sequence = BandSequenceModelModule(
            input_dim_size=fc_dim,
            hidden_dim_size=band_hidden_size,
            rnn_type=rnn_type,
            bidirectional=bidirectional,
            num_layers=band_lstm_layers,
        )
         # Configuration for MaskEstimationModule
        mask_cfg = {
            "sr": sr,
            "n_fft":  n_fft,
            "bandsplits": [
            (1000, 100),
            (4000, 250),
            (8000, 500),
            (16000, 1000),
            (20000, 2000),
        ],
            "t_timesteps": t_timesteps,
            "fc_dim": fc_dim,
            "mlp_dim": mlp_dim,
            "complex_as_channel": complex_as_channel,
            "is_mono": is_mono,
        }
       
        # Step 3: Mask Estimation
        self.mask_estimation = MaskEstimationModule(**mask_cfg)
        
        
        if input_mean is not None:
            input_mean = torch.from_numpy(-input_mean[: self.n_fft]).float()
        else:
            input_mean = torch.zeros(self.n_fft)

        if input_scale is not None:
            input_scale = torch.from_numpy(1.0 / input_scale[: self.n_fft]).float()
        else:
            input_scale = torch.ones(self.n_fft)

        self.input_mean = Parameter(input_mean)
        self.input_scale = Parameter(input_scale)

        self.max_bin= 1500

    def forward(self, x_in: torch.Tensor):
        # permute so that batch is last for lstm
        x_in = x_in.permute(3, 0, 1, 2)
        # get current spectrogram shape
        nb_frames, nb_samples, nb_channels, nb_bins= x_in.data.shape
        # crop
        x_in = x_in[..., : self.n_fft]
        # shift and scale input to mean=0 std=1 (across all bins)
        x_in = x_in + self.input_mean
        x_in = x_in * self.input_scale
        x_in=x_in.permute(1, 2, 3, 0)

        if nb_frames>255:  #validation set
            segment_length=255
            # Pad the track if necessary
            pad_length = (segment_length - (nb_frames % segment_length)) % segment_length
            if pad_length > 0:
                x_in = F.pad(x_in, (0, pad_length))  # Pad time dimension
                nb_frames += pad_length
            output_track = torch.zeros_like(x_in)  # Placeholder for the reconstructed output

            # Process each segment
            for start in range(0, nb_frames, segment_length):
                end = start + segment_length
                segment = x_in[:, :, :, start:end]  # Extract segment
                mask = self.band_split(segment)  # [B, k_subbands, T, fc_dim]
                # Band Sequence Processing
                mask = self.band_sequence(mask)  # [B, k_subbands, T, fc_dim]
                # Mask Estimation and Reconstruction
                mask = self.mask_estimation(mask)  # [B, freq, T]
                output_track[:, :, :, start:end] = mask

            # Remove padding from the output if necessary
            if pad_length > 0:
                output_track = output_track[:, :, :, :-pad_length]
                x_in = x_in[:, :, :, :-pad_length]
            x_out=output_track*x_in
        else:
            # Band Splitting
            mask = self.band_split(x_in)  # [B, k_subbands, T, fc_dim]
            # Band Sequence Processing
            mask = self.band_sequence(mask)  # [B, k_subbands, T, fc_dim]
            # Mask Estimation and Reconstruction
            mask = self.mask_estimation(mask)  # [B, freq, T]
            x_out=mask*x_in
        
        # denormalize
        x_out = x_out.permute(3, 0, 1, 2)
        x_out = (x_out / (self.input_scale + 1e-5))-self.input_mean
        x_out=x_out.permute(1, 2, 3, 0)
        return x_out


class Separator(nn.Module):
    """
    Separator class to encapsulate all the stereo filtering
    as a torch Module, to enable end-to-end learning.

    Args:
        targets (dict of str: nn.Module): dictionary of target models
            the spectrogram models to be used by the Separator.
        niter (int): Number of EM steps for refining initial estimates in a
            post-processing stage. Zeroed if only one target is estimated.
            defaults to `1`.
        residual (bool): adds an additional residual target, obtained by
            subtracting the other estimated targets from the mixture,
            before any potential EM post-processing.
            Defaults to `False`.
        wiener_win_len (int or None): The size of the excerpts
            (number of frames) on which to apply filtering
            independently. This means assuming time varying stereo models and
            localization of sources.
            None means not batching but using the whole signal. It comes at the
            price of a much larger memory usage.
        filterbank (str): filterbank implementation method.
            Supported are `['torch', 'asteroid']`. `torch` is about 30% faster
            compared to `asteroid` on large FFT sizes such as 4096. However,
            asteroids stft can be exported to onnx, which makes is practical
            for deployment.
    """

    def __init__(
        self,
        target_models: Mapping[str, nn.Module],
        niter: int = 0,
        softmask: bool = False,
        residual: bool = False,
        sample_rate: float = 44100.0,
        n_fft: int = 4096,
        n_hop: int = 1024,
        nb_channels: int = 2,
        wiener_win_len: Optional[int] = 300,
        filterbank: str = "torch",
    ):
        super(Separator, self).__init__()

        # saving parameters
        self.niter = niter
        self.residual = residual
        self.softmask = softmask
        self.wiener_win_len = wiener_win_len

        self.stft, self.istft = make_filterbanks(
            n_fft=n_fft,
            n_hop=n_hop,
            center=True,
            method=filterbank,
            sample_rate=sample_rate,
        )
        self.complexnorm = ComplexNorm(mono=nb_channels == 1)

        # registering the targets models
        self.target_models = nn.ModuleDict(target_models)
        # adding till https://github.com/pytorch/pytorch/issues/38963
        self.nb_targets = len(self.target_models)
        # get the sample_rate as the sample_rate of the first model
        # (tacitly assume it's the same for all targets)
        self.register_buffer("sample_rate", torch.as_tensor(sample_rate))

    def freeze(self):
        # set all parameters as not requiring gradient, more RAM-efficient
        # at test time
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, audio: Tensor) -> Tensor:
        """Performing the separation on audio input

        Args:
            audio (Tensor): [shape=(nb_samples, nb_channels, nb_timesteps)]
                mixture audio waveform

        Returns:
            Tensor: stacked tensor of separated waveforms
                shape `(nb_samples, nb_targets, nb_channels, nb_timesteps)`
        """

        nb_sources = self.nb_targets
        nb_samples = audio.shape[0]

        # getting the STFT of mix:
        # (nb_samples, nb_channels, n_fft, nb_frames, 2)
        mix_stft = self.stft(audio)
        X = self.complexnorm(mix_stft)

        # initializing spectrograms variable
        spectrograms = torch.zeros(X.shape + (nb_sources,), dtype=audio.dtype, device=X.device)

        for j, (target_name, target_module) in enumerate(self.target_models.items()):
            # apply current model to get the source spectrogram
            target_spectrogram = target_module(X.detach().clone())
            spectrograms[..., j] = target_spectrogram

        # transposing it as
        # (nb_samples, nb_frames, n_fft,{1,nb_channels}, nb_sources)
        spectrograms = spectrograms.permute(0, 3, 2, 1, 4)

        # rearranging it into:
        # (nb_samples, nb_frames, n_fft, nb_channels, 2) to feed
        # into filtering methods
        mix_stft = mix_stft.permute(0, 3, 2, 1, 4)

        # create an additional target if we need to build a residual
        if self.residual:
            # we add an additional target
            nb_sources += 1

        if nb_sources == 1 and self.niter > 0:
            raise Exception(
                "Cannot use EM if only one target is estimated."
                "Provide two targets or create an additional "
                "one with `--residual`"
            )

        nb_frames = spectrograms.shape[1]
        targets_stft = torch.zeros(
            mix_stft.shape + (nb_sources,), dtype=audio.dtype, device=mix_stft.device
        )
        for sample in range(nb_samples):
            pos = 0
            if self.wiener_win_len:
                wiener_win_len = self.wiener_win_len
            else:
                wiener_win_len = nb_frames
            while pos < nb_frames:
                cur_frame = torch.arange(pos, min(nb_frames, pos + wiener_win_len))
                pos = int(cur_frame[-1]) + 1

                targets_stft[sample, cur_frame] = wiener(
                    spectrograms[sample, cur_frame],
                    mix_stft[sample, cur_frame],
                    self.niter,
                    softmask=self.softmask,
                    residual=self.residual,
                )

        # getting to (nb_samples, nb_targets, channel, fft_size, n_frames, 2)
        targets_stft = targets_stft.permute(0, 5, 3, 2, 1, 4).contiguous()

        # inverse STFT
        estimates = self.istft(targets_stft, length=audio.shape[2])

        return estimates

    def to_dict(self, estimates: Tensor, aggregate_dict: Optional[dict] = None) -> dict:
        """Convert estimates as stacked tensor to dictionary

        Args:
            estimates (Tensor): separated targets of shape
                (nb_samples, nb_targets, nb_channels, nb_timesteps)
            aggregate_dict (dict or None)

        Returns:
            (dict of str: Tensor):
        """
        estimates_dict = {}
        for k, target in enumerate(self.target_models):
            estimates_dict[target] = estimates[:, k, ...]

        # in the case of residual, we added another source
        if self.residual:
            estimates_dict["residual"] = estimates[:, -1, ...]

        if aggregate_dict is not None:
            new_estimates = {}
            for key in aggregate_dict:
                new_estimates[key] = torch.tensor(0.0)
                for target in aggregate_dict[key]:
                    new_estimates[key] = new_estimates[key] + estimates_dict[target]
            estimates_dict = new_estimates
        return estimates_dict
