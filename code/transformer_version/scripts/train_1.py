import argparse
import torch
import time
from pathlib import Path
import tqdm
import json
import sklearn.preprocessing
import numpy as np
import random
from git import Repo
import os
import copy
import torchaudio
import logging
import matplotlib.pyplot as plt

from openunmix1 import data
from openunmix1 import model
from openunmix1 import utils
from openunmix1 import transforms

tqdm.monitor_interval = 0


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

"""
def train(args, unmix, encoder, device, train_sampler, optimizer):
    losses = utils.AverageMeter()
    unmix.train()
    pbar = tqdm.tqdm(train_sampler, disable=args.quiet)
    for x, y in pbar:
        pbar.set_description("Training batch")
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        X = encoder(x)
        Y_hat = unmix(X)
        Y = encoder(y)
        loss = torch.nn.functional.mse_loss(Y_hat, Y)
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), Y.size(1))
        pbar.set_postfix(loss="{:.3f}".format(losses.avg))
    # Clear memory after training
    torch.cuda.empty_cache()  
    return losses.avg



def valid(args, unmix, encoder, device, valid_sampler):
    losses = utils.AverageMeter()
    unmix.eval()
    with torch.no_grad():
        for x, y in valid_sampler:
            x, y = x.to(device), y.to(device)
            X = encoder(x)
            X.requires_grad_(True)
            Y_hat = unmix(X)
            X.requires_grad_(False)
            torch.cuda.empty_cache()
            Y = encoder(y)
            torch.cuda.empty_cache()
            loss = torch.nn.functional.mse_loss(Y_hat, Y)
            losses.update(loss.item(), Y.size(1))
        torch.cuda.empty_cache()  
        return losses.avg
"""
def train(args, unmix, encoder, device, train_sampler, optimizer, accumulation_steps):
    losses = utils.AverageMeter()
    unmix.train()
    pbar = tqdm.tqdm(train_sampler, disable=args.quiet)

    for i, (x, y) in enumerate(pbar):
        pbar.set_description("Training batch")
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        # Forward pass
        X = encoder(x)
        Y_hat = unmix(X)
        Y = encoder(y)
        loss = torch.nn.functional.mse_loss(Y_hat, Y)
        loss = loss / accumulation_steps  # Scale the loss for gradient accumulation
        
        # Backward pass
        loss.backward()  # No scaling needed since we're not using mixed precision
        
        # Update the model parameters every accumulation_steps
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()  # Step the optimizer
            optimizer.zero_grad()  # Clear gradients for the next step
            


        losses.update(loss.item() * accumulation_steps, Y.size(1))
        pbar.set_postfix(loss="{:.3f}".format(losses.avg))
     # Turn off interactive mode and display the final plot
    plt.ioff()
    plt.show()

    return losses.avg


def valid(args, unmix, encoder, device, valid_sampler):
    losses = utils.AverageMeter()
    unmix.eval()
    with torch.no_grad():
        for x, y in valid_sampler:
            x, y = x.to(device), y.to(device)
            # Forward pass
            X = encoder(x)
            Y_hat = unmix(X)
            Y = encoder(y)
            loss = torch.nn.functional.mse_loss(Y_hat, Y)
            losses.update(loss.item(), Y.size(1))

        # Clear memory after validation
        torch.cuda.empty_cache()  
        
    return losses.avg


def get_statistics(args, encoder, dataset):
    encoder = copy.deepcopy(encoder).to("cpu")
    scaler = sklearn.preprocessing.StandardScaler()

    dataset_scaler = copy.deepcopy(dataset)
    if isinstance(dataset_scaler, data.SourceFolderDataset):
        dataset_scaler.random_chunks = False
    else:
        dataset_scaler.random_chunks = False
        dataset_scaler.seq_duration = None

    dataset_scaler.samples_per_track = 1
    dataset_scaler.augmentations = None
    dataset_scaler.random_track_mix = False
    dataset_scaler.random_interferer_mix = False

    pbar = tqdm.tqdm(range(len(dataset_scaler)), disable=args.quiet)
    for ind in pbar:
        x, y = dataset_scaler[ind]
        pbar.set_description("Compute dataset statistics")
        # downmix to mono channel
        X = encoder(x[None, ...]).mean(1, keepdim=False).permute(0, 2, 1)

        scaler.partial_fit(np.squeeze(X))

    # set inital input scaler values
    std = np.maximum(scaler.scale_, 1e-4 * np.max(scaler.scale_))
    return scaler.mean_, std
"""
class WarmupLinearSchedule(torch.optim.lr_scheduler.LambdaLR):
        def __init__(self, optimizer, warmup_steps, total_steps):
            self.warmup_steps = warmup_steps
            self.total_steps = total_steps
            super().__init__(optimizer, self.get_lr_lambda)

        def get_lr_lambda(self, step):
            if step < self.warmup_steps:
                return float(step) / float(max(1, self.warmup_steps))
            return max(0.0, float(self.total_steps - step) / float(max(1, self.total_steps - self.warmup_steps)))
"""
    
def main():
    parser = argparse.ArgumentParser(description="Open Unmix Trainer")

    # which target do we want to train?
    parser.add_argument(
        "--target",
        type=str,
        default="vocals",
        help="target source (will be passed to the dataset)",
    )

    # Dataset paramaters
    parser.add_argument(
        "--dataset",
        type=str,
        default="musdb",
        choices=[
            "musdb",
            "aligned",
            "sourcefolder",
            "trackfolder_var",
            "trackfolder_fix",
        ],
        help="Name of the dataset.",
    )
    parser.add_argument("--root", type=str, help="root path of dataset")
    parser.add_argument(
        "--output",
        type=str,
        default="open-unmix",
        help="provide output path base folder name",
    )
    parser.add_argument("--model", type=str, help="Name or path of pretrained model to fine-tune")
    parser.add_argument("--checkpoint", type=str, help="Path of checkpoint to resume training")
    parser.add_argument(
        "--audio-backend",
        type=str,
        default="soundfile",
        help="Set torchaudio backend (`sox_io` or `soundfile`",
    )

    # Training Parameters
    parser.add_argument("--epochs", type=int, default=1000)                                 #epochs and batch
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate, defaults to 1e-3")
    parser.add_argument(
        "--patience",
        type=int,
        default=240,
        help="maximum number of train epochs (default: 140)",
    )
    parser.add_argument(
        "--lr-decay-patience",
        type=int,
        default=80,
        help="lr decay patience for plateau scheduler",
    )
    parser.add_argument(
        "--lr-decay-gamma",
        type=float,
        default=0.3,
        help="gamma of learning rate scheduler decay",
    )
    parser.add_argument("--weight-decay", type=float, default=0.00001, help="weight decay")
    parser.add_argument(
        "--seed", type=int, default=42, metavar="S", help="random seed (default: 42)"
    )

    # Model Parameters
    parser.add_argument(
        "--seq-dur",
        type=float,
        default=6.0,
        help="Sequence duration in seconds" "value of <=0.0 will use full/variable length",
    )
    parser.add_argument(
        "--unidirectional",
        action="store_true",
        default=False,
        help="Use unidirectional LSTM",
    )
    parser.add_argument("--nfft", type=int, default=4096, help="STFT fft size and window size")
    parser.add_argument("--nhop", type=int, default=1024, help="STFT hop size")
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=512,
        help="hidden size parameter of bottleneck layers",
    )
    parser.add_argument(
        "--bandwidth", type=int, default=16000, help="maximum model bandwidth in herz"
    )
    parser.add_argument(
        "--nb-channels",
        type=int,
        default=2,
        help="set number of channels for model (1, 2)",
    )
    parser.add_argument(
        "--nb-workers", type=int, default=6, help="Number of workers for dataloader."                                     #workers
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Speed up training init for dev purposes",
    )

    # Misc Parameters
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="less verbose during training",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )

    args, _ = parser.parse_known_args()

    torchaudio.set_audio_backend(args.audio_backend)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print("Using GPU:", use_cuda)
    dataloader_kwargs = {"num_workers": args.nb_workers, "pin_memory": True} if use_cuda else {}

    #repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    #repo = Repo(repo_dir)
    #commit = repo.head.commit.hexsha[:7]

    # use jpg or npy
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_dataset, valid_dataset, args = data.load_datasets(parser, args)

    # create output dir if not exist
    target_path = Path(args.output)
    target_path.mkdir(parents=True, exist_ok=True)

    train_sampler = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, **dataloader_kwargs
    )
    valid_sampler = torch.utils.data.DataLoader(valid_dataset, batch_size=1, **dataloader_kwargs)

    stft, _ = transforms.make_filterbanks(
        n_fft=args.nfft, n_hop=args.nhop, sample_rate=train_dataset.sample_rate
    )
    encoder = torch.nn.Sequential(stft, model.ComplexNorm(mono=args.nb_channels == 1)).to(device)

    separator_conf = {
        "nfft": args.nfft,
        "nhop": args.nhop,
        "sample_rate": train_dataset.sample_rate,
        "nb_channels": args.nb_channels,
    }

    with open(Path(target_path, "separator.json"), "w") as outfile:
        outfile.write(json.dumps(separator_conf, indent=4, sort_keys=True))

    if args.checkpoint or args.model or args.debug:
        scaler_mean = None
        scaler_std = None
    else:
        scaler_mean, scaler_std = get_statistics(args, encoder, train_dataset)

    max_bin = utils.bandwidth_to_max_bin(train_dataset.sample_rate, args.nfft, args.bandwidth)

    if args.model:
        # fine tune model
        print(f"Fine-tuning model from {args.model}")
        unmix = utils.load_target_models(
            args.target, model_str_or_path=args.model, device=device, pretrained=True
        )[args.target]
        unmix = unmix.to(device)
    else:
        unmix = model.OpenUnmix(
            input_mean=scaler_mean,
            input_scale=scaler_std,
            nb_bins=args.nfft // 2 + 1,
            nb_channels=args.nb_channels,
            hidden_size=args.hidden_size,
            max_bin=max_bin,
            unidirectional=args.unidirectional
        ).to(device)
    
    optimizer = torch.optim.Adam(unmix.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=args.lr_decay_gamma,
        patience=args.lr_decay_patience,
        cooldown=10,
    )
    
# MY CHANGE @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    """
    # Define the optimizer
    optimizer = torch.optim.Adam(unmix.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Set up the scheduler
    total_steps = args.epochs * len(train_sampler)  # Total training steps
    warmup_steps = int(0.05 * total_steps)  # 10% of total steps for warmup
    logging.info(f'Warmup Steps: {warmup_steps}')
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps, total_steps)
    """
    es = utils.EarlyStopping(patience=args.patience)

    # if a checkpoint is specified: resume training
    if args.checkpoint:
        model_path = Path(args.checkpoint).expanduser()
        with open(Path(model_path, args.target + ".json"), "r") as stream:
            results = json.load(stream)

        target_model_path = Path(model_path, args.target + ".chkpnt")
        checkpoint = torch.load(target_model_path, map_location=device)
        unmix.load_state_dict(checkpoint["state_dict"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        # train for another epochs_trained
        t = tqdm.trange(
            results["epochs_trained"],
            results["epochs_trained"] + args.epochs + 1,
            disable=args.quiet,
        )
        train_losses = results["train_loss_history"]
        valid_losses = results["valid_loss_history"]
        train_times = results["train_time_history"]
        best_epoch = results["best_epoch"]
        es.best = results["best_loss"]
        es.num_bad_epochs = results["num_bad_epochs"]
    # else start optimizer from scratch
    else:
        t = tqdm.trange(1, args.epochs + 1, disable=args.quiet)
        train_losses = []
        valid_losses = []
        train_times = []
        best_epoch = 0
    
    accumulation_steps = 4  # Set the number of steps to accumulate gradients

    for epoch in t:
        t.set_description("Training epoch")
        end = time.time()
        #print("before training:")
        #print(torch.cuda.memory_summary(device=None, abbreviated=False))
        train_loss = train(args, unmix, encoder, device, train_sampler, optimizer,accumulation_steps)
        #print("after training:")
        #print(torch.cuda.memory_summary(device=None, abbreviated=False))
        valid_loss = valid(args, unmix, encoder, device, valid_sampler)
        #print("after valid:")
        #print(torch.cuda.memory_summary(device=None, abbreviated=False))
        # Free up memory after validation
        torch.cuda.empty_cache()
        scheduler.step(valid_loss)
        logging.info(f'Epoch {epoch}: Train Loss: {train_loss:.3f}, Validation Loss: {valid_loss:.3f}')
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        t.set_postfix(train_loss=train_loss, val_loss=valid_loss)

        stop = es.step(valid_loss)

        if valid_loss == es.best:
            best_epoch = epoch

        utils.save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": unmix.state_dict(),
                "best_loss": es.best,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            is_best=valid_loss == es.best,
            path=target_path,
            target=args.target,
        )

        # save params
        params = {
            "epochs_trained": epoch,
            "args": vars(args),
            "best_loss": es.best,
            "best_epoch": best_epoch,
            "train_loss_history": train_losses,
            "valid_loss_history": valid_losses,
            "train_time_history": train_times,
            "num_bad_epochs": es.num_bad_epochs,
            #"commit": commit,
        }

        with open(Path(target_path, args.target + ".json"), "w") as outfile:
            outfile.write(json.dumps(params, indent=4, sort_keys=True))

        train_times.append(time.time() - end)

        # Print alpha and beta values at the end of each epoch
        #print(f"Alpha (LSTM): {unmix.music_transformer.alpha.item():.4f}")
        #print(f"Beta (Transformer): {unmix.music_transformer.beta.item():.4f}")

        if stop:
            print("Apply Early Stopping")
            break


if __name__ == "__main__":
    main()
