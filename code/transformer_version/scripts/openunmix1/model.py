from typing import Optional, Mapping

import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch.nn import LSTM, BatchNorm1d, Linear, Parameter
from .filtering import wiener
from .transforms import make_filterbanks, ComplexNorm

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()


        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


class VocalAttention(nn.Module):
    def __init__(self, hidden_size: int, num_bins: int):
        super(VocalAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_bins = num_bins
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        # Define vocal range bins
        self.vocal_range_bins = [
            1,2,3,4,5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
            21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
            33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
            45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
            57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68,
            69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80
        ]
        
        # Initialize the vocal mask
        vocal_mask = torch.randn(1, 1,num_bins) * 0.1

        # Set specific vocal frequency bins to a higher initial value
        for bin_index in self.vocal_range_bins:
            if bin_index < num_bins:
                vocal_mask[0, 0, bin_index] = 1.0
         
        # Make vocal_mask a parameter
        self.vocal_mask = nn.Parameter(vocal_mask)

    def forward(self, x):
        query = self.query(x)  # Shape: (nb_frames, batch_size, hidden_size)
        key = self.key(x)      # Shape: (nb_frames, batch_size, hidden_size)
        value = self.value(x)  # Shape: (nb_frames, batch_size, hidden_size)

        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.hidden_size ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        
        # Compute attended values using attention weights
        attn_output = torch.matmul(attn_weights, value) # Shape: (nb_frames, batch_size, hidden_size)

        # Apply vocal mask over the frequency bins
        # Reshape vocal_mask to match the (batch_size, hidden_size) dimensions
        # Ensure that attn_output and vocal_mask have compatible shapes
        attn_output = attn_output * self.vocal_mask[:, :, :attn_output.size(-1)]  # Apply only up to attn_output size

        #print("attn_output shape:", attn_output.shape)  # Shape should be (nb_frames, batch_size, hidden_size)
        #print("vocal_mask shape:", self.vocal_mask.shape)  # Shape should be (1, 1, num_bins)
        
        return attn_output

class VocalAttentionNew(nn.Module):
    def __init__(self, nb_bins, hidden_size, vocal_range_bins=None):
        """
        Args:
            nb_bins (int): Total number of frequency bins in the spectrogram.
            hidden_size (int): The size of the hidden state for the attention mechanism.
            vocal_range_bins (list[int], optional): Indices of frequency bins related to vocals.
                If None, no specific vocal range masking is applied.
        """
        super(VocalAttentionNew, self).__init__()
        self.nb_bins = nb_bins
        self.hidden_size = hidden_size
        self.vocal_range_bins = vocal_range_bins or []

        # Linear transformations for query, key, and value
        self.query = nn.Linear(nb_bins, hidden_size)
        self.key = nn.Linear(nb_bins, hidden_size)
        self.value = nn.Linear(nb_bins, hidden_size)

        # Linear projection to restore original number of bins
        self.projection = nn.Linear(hidden_size, nb_bins)

        # Vocal frequency mask
        self.vocal_mask = nn.Parameter(torch.ones(1, 1, nb_bins))
        if vocal_range_bins:
            mask = torch.ones(1, 1, nb_bins)*0.25
            for bin_idx in vocal_range_bins:
                if bin_idx < nb_bins:
                    mask[0, 0, bin_idx] = 1.0
            self.vocal_mask.data = mask

    def forward(self, x):
        """
        Args:
            x (Tensor): Input spectrogram of shape `(nb_frames, nb_samples, nb_channels, nb_bins)`.

        Returns:
            Tensor: Spectrogram with attention applied, same shape as input.
        """
        # Reshape for attention: merge channels into samples
        nb_frames, nb_samples, nb_channels, nb_bins = x.shape
        x = x.view(nb_frames, nb_samples * nb_channels, nb_bins)

        # Apply vocal mask *before* the attention process
        x = x * self.vocal_mask

        # Linear transformations
        query = self.query(x)  # Shape: (nb_frames, nb_samples * nb_channels, hidden_size)
        key = self.key(x)      # Shape: (nb_frames, nb_samples * nb_channels, hidden_size)
        value = self.value(x)  # Shape: (nb_frames, nb_samples * nb_channels, hidden_size)

        # Compute scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.hidden_size ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, value)  # Shape: (nb_frames, nb_samples * nb_channels, hidden_size)

        # Restore the original number of bins
        x = self.projection(attn_output)  # Shape: (nb_frames, nb_samples * nb_channels, nb_bins)

        # Restore original channel dimension
        x = x.view(nb_frames, nb_samples, nb_channels, nb_bins)
        return x
    
class MusicTransformer(nn.Module):
    def __init__(self, hidden_size, num_heads, num_layers, num_bins, dropout):
        super(MusicTransformer, self).__init__()
        #self.input_conv = nn.Conv1d(in_channels=1, out_channels=hidden_size, kernel_size=3, padding=1)
        self.positional_encoding = PositionalEncoding(hidden_size)
        self.vocal_attention = VocalAttention(hidden_size, num_bins)
        # Create the transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        #x = self.input_conv(x)  # Applying convolution

        # x shape: (nb_frames, batch_size, hidden_size)
        x = self.positional_encoding(x)  # Add positional encoding
        
        # Apply vocal attention before transformer layers
        x = self.vocal_attention(x)
        
        # Pass through the transformer encoder
        x = self.transformer(x)  # Output shape will be (nb_frames, batch_size, hidden_size)

        
        
        return self.norm(x)


class SimpleTransformer(nn.Module):
    def __init__(self, hidden_size, num_heads, num_layers, dropout):
        super(SimpleTransformer, self).__init__()

        # Define a transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dropout=dropout)
        
        # Create the transformer encoder
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        #self.input_conv = nn.Conv1d(in_channels=1, out_channels=hidden_size, kernel_size=3, padding=1)
        self.positional_encoding = PositionalEncoding(hidden_size)  # Ensure this is defined

    def forward(self, x):
        # x shape: (nb_frames, batch_size, hidden_size)

        # No need for input convolution since x is already in (nb_frames, batch_size, hidden_size)
        # If you decide to apply a convolution, ensure it's compatible with this shape
        
        # Add positional encoding
        x = self.positional_encoding(x)  # Assuming this function can handle the shape (nb_frames, batch_size, hidden_size)

        # Pass through the transformer encoder
        x = self.transformer(x)  # Output shape will be (nb_frames, batch_size, hidden_size)

        return x  # Or apply further processing as needed

class HybridMusicTransformer(nn.Module):
    def __init__(self, hidden_size, num_layers, num_heads, num_bins,dropout):
        super().__init__()
        self.positional_encoding = PositionalEncoding(hidden_size)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                dropout=dropout,
                activation='relu',
            ),
            num_layers=num_layers,
        )
        self.lstm = LSTM(hidden_size, hidden_size//2, num_layers, batch_first=False,bidirectional=True,dropout=0.4)
        
        # Learnable weighting parameters
        self.alpha = nn.Parameter(torch.tensor(0.6))  #lstm
        self.beta = nn.Parameter(torch.tensor(0.4)) #transformerEncoder
        
        # Layer normalization for each module
        self.lstm_norm = nn.LayerNorm(hidden_size)
        self.encoder_norm = nn.LayerNorm(hidden_size)
        
        # Final normalization
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        # Add positional encoding
        x = self.positional_encoding(x)

        # LSTM layer
        lstm_out, _ = self.lstm(x)
        lstm_out = self.lstm_norm(lstm_out)

        # Transformer Encoder
        transformer_out = self.encoder(x)
        transformer_out = self.encoder_norm(transformer_out)

        # Normalize weights to ensure they sum to 1
        total = self.alpha + self.beta
        alpha_norm = self.alpha / total
        beta_norm = self.beta / total

        # Weighted combination of outputs
        x = alpha_norm * lstm_out + beta_norm * transformer_out

        # Final normalization
        return self.norm(x)

def create_2d_sin_embedding(d_model, height, width, device="cpu", max_period=10000):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError(
            "Cannot use sin/cos positional encoding with "
            "odd dimension (got dim={:d})".format(d_model)
        )
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(
        torch.arange(0.0, d_model, 2) * -(math.log(max_period) / d_model)
    )
    pos_w = torch.arange(0.0, width).unsqueeze(1)
    pos_h = torch.arange(0.0, height).unsqueeze(1)
    pe[0:d_model:2, :, :] = (
        torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    )
    pe[1:d_model:2, :, :] = (
        torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    )
    pe[d_model::2, :, :] = (
        torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    )
    pe[d_model + 1:: 2, :, :] = (
        torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    )

    return pe[None, :].to(device)

class ScaledEmbedding(nn.Module):
    """
    Boost learning rate for embeddings (with `scale`).
    Also, can make embeddings continuous with `smooth`.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int,
                 scale: float = 10., smooth=True):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        if smooth:
            weight = torch.cumsum(self.embedding.weight.data, dim=0)
            # when summing gaussian, overscale raises as sqrt(n), so we nornalize by that.
            weight = weight / torch.arange(1, num_embeddings + 1).to(weight).sqrt()[:, None]
            self.embedding.weight.data[:] = weight
        self.embedding.weight.data /= scale
        self.scale = scale

    @property
    def weight(self):
        return self.embedding.weight * self.scale

    def forward(self, x):
        out = self.embedding(x) * self.scale
        return out

class OpenUnmix(nn.Module):    # problem: in validation you dont get 6 sec segments like in train.
    def __init__(
        self,
        nb_bins=2048,
        nb_channels=2,
        hidden_size=300,
        nb_layers=4,
        num_heads=1,
        dropout=0.2,
        reduction_factor=5 * 5 * 4 * 6,
        unidirectional=False,
        input_mean=None,
        input_scale=None,
        max_bin=None,
    ):
        super(OpenUnmix, self).__init__()

        self.hidden_size = 300
        self.nb_channels = nb_channels
        self.reduction_factor = reduction_factor
        self.nb_output_bins = nb_bins
        if max_bin:
            self.nb_bins = 1800
            self.nb_output_bins=1800
        else:
            self.nb_bins = self.nb_output_bins

        # Frequency Encoder
        self.freq_encoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(nb_channels, 48, kernel_size=(3, 3), padding=(1, 1)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(6, 1), stride=(6, 1))
            ),
            nn.Sequential(
                nn.Conv2d(48, 96, kernel_size=(3, 3), padding=(1, 1)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(5, 1), stride=(5, 1))
            ),
            nn.Sequential(
                nn.Conv2d(96, 192, kernel_size=(3, 3), padding=(1, 1)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(5, 1), stride=(5, 1))
            ),
            nn.Sequential(
                nn.Conv2d(192, self.hidden_size, kernel_size=(3, 3), padding=(1, 1)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1))
            ),
        ])

        # Time Encoder
        self.time_encoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(nb_channels, 48, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=6, stride=6)
            ),
            nn.Sequential(
                nn.Conv1d(48, 96, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=5, stride=5)
            ),
            nn.Sequential(
                nn.Conv1d(96, 192, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=5, stride=5)
            ),
            nn.Sequential(
                nn.Conv1d(192, self.hidden_size, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=4, stride=4)
            ),
        ])

        self.freq_projector = nn.Linear(self.hidden_size* (self.nb_bins//self.reduction_factor), self.hidden_size)
        self.time_projector = nn.Linear(self.hidden_size, self.hidden_size)

        # Cross-Attention Transformer
        self.cross_attention = nn.Transformer(
            d_model=self.hidden_size ,                           #needs to be hidden size, build linear layers to reach this.
            nhead=num_heads,
            num_encoder_layers=nb_layers,
            num_decoder_layers=nb_layers,
            dim_feedforward=self.hidden_size * 4,
            dropout=dropout,
            batch_first=True,
        )

        #self.freq_projector_decoder_bins = nn.Linear(self.nb_bins_time//self.reduction_factor, 255)
        self.freq_projector_decoder_channel = nn.Linear(self.hidden_size, self.hidden_size*(self.nb_bins//self.reduction_factor))
        # Frequency Embedding
        self.freq_emb = ScaledEmbedding(nb_bins, embedding_dim=self.hidden_size)

        # Frequency Decoder
        self.freq_decoder = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(self.hidden_size, 192, kernel_size=(4, 1), stride=(4, 1), padding=(0, 0)),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.ConvTranspose2d(192, 96, kernel_size=(5, 1), stride=(5, 1), padding=(0, 0)),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.ConvTranspose2d(96 , 48, kernel_size=(5, 1), stride=(5, 1), padding=(0, 0)),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.ConvTranspose2d(48 , nb_channels, kernel_size=(6, 1), stride=(6, 1), padding=(0, 0))
            ),
        ])

        # Time Decoder
        self.time_decoder = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose1d(self.hidden_size, 192, kernel_size=4, stride=4, padding=0),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.ConvTranspose1d( 192, 96, kernel_size=5, stride=5, padding=0),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.ConvTranspose1d(96, 48, kernel_size=5, stride=5, padding=0),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.ConvTranspose1d(48, nb_channels, kernel_size=6, stride=6, padding=0)
            ),
        ])
        

        if input_mean is not None:
            input_mean = torch.from_numpy(-input_mean[: self.nb_bins]).float()
        else:
            input_mean = torch.zeros(self.nb_bins)

        if input_scale is not None:
            input_scale = torch.from_numpy(1.0 / input_scale[: self.nb_bins]).float()
        else:
            input_scale = torch.ones(self.nb_bins)

        self.input_mean = Parameter(input_mean)
        self.input_scale = Parameter(input_scale)
        self.output_scale = Parameter(torch.ones(self.nb_output_bins).float())
        self.output_mean = Parameter(torch.ones(self.nb_output_bins).float())
        
        self.stft, _ = make_filterbanks(
        n_fft=4096, n_hop=1024, sample_rate=44100
    )
        self.encoder = torch.nn.Sequential(self.stft, ComplexNorm(mono=False))
    def forward(self, x, xt):
        """
        Forward pass of the model.
        x: Input spectrogram tensor with shape (B, C, F, T) (frequency domain)
        xt: Input tensor with shape (B, C, T) (time domain)
        """
        B, C, Ft, T = x.shape
        x = x[:, :, : self.nb_bins, :]
        x = x.permute(3, 0, 1, 2)
        x = x + self.input_mean
        x = x * self.input_scale
        x = x.permute(1, 2, 3, 0)

         # Prepare the time branch input.
        meant = xt.mean(dim=(1, 2), keepdim=True)
        stdt = xt.std(dim=(1, 2), keepdim=True)
        xt = (xt - meant) / (1e-5 + stdt)
        """
        print("x shape ")
        print(x.shape)
        print("xt shape ")
        print(xt.shape)
        """
       # Frequency Encoding with skip connections
        freq_encoded_stages = []
        freq_encoded = x
        for layer in self.freq_encoder:
            freq_encoded = layer(freq_encoded)
            freq_encoded_stages.append(freq_encoded)

        reduced_F = self.nb_bins // self.reduction_factor
        frs = torch.arange(reduced_F, device=x.device)  # Frequency bin indices
        freq_emb = self.freq_emb(frs).t()[None, :, :, None].expand(B, self.hidden_size, reduced_F, T)
        freq_encoded = freq_encoded + freq_emb

        # Time Encoding with skip connections
        time_encoded_stages = []
        time_encoded = xt
        for layer in self.time_encoder:
            time_encoded = layer(time_encoded)
            time_encoded_stages.append(time_encoded)
        """
        print("freq_encoded shape 0")
        print(freq_encoded.shape)
        print("time_encoded shape 0")
        print(time_encoded.shape)
        """
        # Cross-Attention
        freq_encoded_flat = rearrange(freq_encoded, "B C F T -> B T (F C)")
        time_encoded_flat = rearrange(time_encoded, "B C T -> B T C")
        """
        print("freq_encoded_flat shape 1")
        print(freq_encoded_flat.shape)
        print("time_encoded_flat shape 1")
        print(time_encoded_flat.shape)
        """
        freq_encoded_flat_proj = self.freq_projector(freq_encoded_flat)
        time_encoded_flat_proj = self.time_projector(time_encoded_flat)
        """
        print("freq_encoded_flat_proj shape 2")
        print(freq_encoded_flat_proj.shape)
        print("time_encoded_flat_proj shape 2")
        print(time_encoded_flat_proj.shape)
        """
        #add positional encoding

        cross_out = self.cross_attention(
            src=freq_encoded_flat_proj, tgt=time_encoded_flat_proj
        )

        #print("cross_out shape 0")
        #print(cross_out.shape)               # B T C ([2, 441, 384])
       
        # Rearrange cross_out for time-domain decoding
        cross_out_time = rearrange(cross_out, "B T C -> B C T")  # (B, C, T) -> ([2, 384, 441])

        # Project cross_out for frequency domain decoding
        cross_out_freq = self.freq_projector_decoder_channel(cross_out)  # (B, T, C) -> (B, T, 768)
        cross_out_freq = rearrange(cross_out_freq, "B T C -> B C T")  # (B, T, 768) -> (B, 768, T)

        # Interpolate cross_out_freq to match the target size T (from x.shape)
        cross_out_freq = F.interpolate(
            cross_out_freq,  # Ensure this is in the form (B, C, T)
            size=T,  # Target time dimension
            mode="linear",
            align_corners=False
        )

        cross_out_freq = rearrange(cross_out_freq, "B (C F) T -> B C F T", C=self.hidden_size, F=reduced_F)         # B 768 255  to  B 255 768  

        
        """
        print("cross_out_freq shape 1")
        print(cross_out_freq.shape)

        print("cross_out_time shape 1")
        print(cross_out_time.shape)
        """
        # Frequency decoding
        for layer in self.freq_decoder:
            cross_out_freq = layer(cross_out_freq)

        # Time decoding
        for layer in self.time_decoder:
            cross_out_time = layer(cross_out_time)
        """
        print("freq_decoded shape ")
        print(cross_out_freq.shape)

        print("time_decoded shape ")
        print(cross_out_time.shape)
        """
        cross_out_time_stft=self.encoder(cross_out_time)
        #print("cross_out_time_stft shape ")
        #print(cross_out_time_stft.shape)
        diff = cross_out_freq.shape[3] - cross_out_time_stft.shape[3]
        if diff > 0:
            cross_out_time_stft = F.pad(cross_out_time_stft, (0, diff))  # Pad on the right
        elif diff < 0:
            cross_out_freq = F.pad(cross_out_freq, (0, -diff))  # Pad on the right
        output=cross_out_freq + cross_out_time_stft[:,:,:1800,:]
        output = output.permute(3, 0, 1, 2)
        output = (output - self.output_mean) * self.output_scale

        return output.permute(1, 2, 3, 0)


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
        target_models: dict,
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
        # (nb_samples, nb_channels, nb_bins, nb_frames, 2)
        mix_stft = self.stft(audio)
        X = self.complexnorm(mix_stft)

        # initializing spectrograms variable
        spectrograms = torch.zeros(X.shape + (nb_sources,), dtype=audio.dtype, device=X.device)

        for j, (target_name, target_module) in enumerate(self.target_models.items()):
            # apply current model to get the source spectrogram
            target_spectrogram = target_module(X.detach().clone())
            spectrograms[..., j] = target_spectrogram

        # transposing it as
        # (nb_samples, nb_frames, nb_bins,{1,nb_channels}, nb_sources)
        spectrograms = spectrograms.permute(0, 3, 2, 1, 4)

        # rearranging it into:
        # (nb_samples, nb_frames, nb_bins, nb_channels, 2) to feed
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
