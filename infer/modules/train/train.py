import os
import sys
import logging

logger = logging.getLogger(__name__)

now_dir = os.getcwd()
sys.path.append(os.path.join(now_dir))

import datetime

from infer.lib.train import utils

hps = utils.get_hparams()
os.environ["CUDA_VISIBLE_DEVICES"] = hps.gpus.replace("-", ",")
n_gpus = len(hps.gpus.split("-"))
from random import randint, shuffle

import torch

try:
    import intel_extension_for_pytorch as ipex  # pylint: disable=import-error, unused-import

    if torch.xpu.is_available():
        from infer.modules.ipex import ipex_init
        from infer.modules.ipex.gradscaler import gradscaler_init
        from torch.xpu.amp import autocast

        GradScaler = gradscaler_init()
        ipex_init()
    else:
        from torch.cuda.amp import GradScaler, autocast
except Exception:
    from torch.cuda.amp import GradScaler, autocast

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
from time import sleep
from time import time as ttime

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from infer.lib.infer_pack import commons
from infer.lib.train.data_utils import (
    DistributedBucketSampler,
    TextAudioCollate,
    TextAudioCollateMultiNSFsid,
    TextAudioLoader,
    TextAudioLoaderMultiNSFsid,
)

if hps.version == "v1":
    from infer.lib.infer_pack.models import MultiPeriodDiscriminator
    from infer.lib.infer_pack.models import SynthesizerTrnMs256NSFsid as RVC_Model_f0
    from infer.lib.infer_pack.models import (
        SynthesizerTrnMs256NSFsid_nono as RVC_Model_nof0,
    )
else:
    from infer.lib.infer_pack.models import (
        SynthesizerTrnMs768NSFsid as RVC_Model_f0,
        SynthesizerTrnMs768NSFsid_nono as RVC_Model_nof0,
        MultiPeriodDiscriminatorV2 as MultiPeriodDiscriminator,
    )

from infer.lib.train.losses import (
    discriminator_loss,
    feature_loss,
    generator_loss,
    kl_loss,
)
from infer.lib.train.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from infer.lib.train.process_ckpt import savee

global_step = 0


class EpochRecorder:
    def __init__(self):
        self.last_time = ttime()

    def record(self):
        now_time = ttime()
        elapsed_time = now_time - self.last_time
        self.last_time = now_time
        elapsed_time_str = str(datetime.timedelta(seconds=elapsed_time))
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"[{current_time}] | ({elapsed_time_str})"


def main():
    n_gpus = torch.cuda.device_count()

    if torch.cuda.is_available() == False and torch.backends.mps.is_available() == True:
        n_gpus = 1
    if n_gpus < 1:
        print("NO GPU DETECTED: falling back to CPU - this may take a while")
        n_gpus = 1
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(randint(20000, 55555))
    children = []
    logger = utils.get_logger(hps.model_dir)
    for i in range(n_gpus):
        subproc = mp.Process(
            target=run,
            args=(i, n_gpus, hps, logger),
        )
        children.append(subproc)
        subproc.start()

    for i in range(n_gpus):
        children[i].join()


def run(rank, n_gpus, hps, logger: logging.Logger):
    global global_step
    if rank == 0:
        logger.info(f"Hyperparameters: {hps}")
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    dist.init_process_group(
        backend="gloo", init_method="env://", world_size=n_gpus, rank=rank
    )
    torch.manual_seed(hps.train.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    if hps.if_f0 == 1:
        train_dataset = TextAudioLoaderMultiNSFsid(hps.data.training_files, hps.data)
    else:
        train_dataset = TextAudioLoader(hps.data.training_files, hps.data)
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size * n_gpus,
        [100, 200, 300, 400, 500, 600, 700, 800, 900],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
    )
    if hps.if_f0 == 1:
        collate_fn = TextAudioCollateMultiNSFsid()
    else:
        collate_fn = TextAudioCollate()
    train_loader = DataLoader(
        train_dataset,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
        persistent_workers=True,
        prefetch_factor=8,
    )
    if hps.if_f0 == 1:
        net_g = RVC_Model_f0(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model,
            is_half=hps.train.fp16_run,
            sr=hps.sample_rate,
        )
    else:
        net_g = RVC_Model_nof0(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model,
            is_half=hps.train.fp16_run,
        )
    if torch.cuda.is_available():
        net_g = net_g.cuda(rank)
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm)
    if torch.cuda.is_available():
        net_d = net_d.cuda(rank)
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        pass
    elif torch.cuda.is_available():
        net_g = DDP(net_g, device_ids=[rank])
        net_d = DDP(net_d, device_ids=[rank])
    else:
        net_g = DDP(net_g)
        net_d = DDP(net_d)

    try:
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d
        )
        if rank == 0:
            logger.info(f"Loaded Discriminator checkpoint at epoch {epoch_str}")
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g
        )
        global_step = (epoch_str - 1) * len(train_loader)
    except:
        epoch_str = 1
        global_step = 0
        if hps.pretrainG != "":
            if rank == 0:
                logger.info(f"Loaded pretrained generator from {hps.pretrainG}")
            if hasattr(net_g, "module"):
                logger.info(
                    net_g.module.load_state_dict(
                        torch.load(hps.pretrainG, map_location="cpu")["model"]
                    )
                )
            else:
                logger.info(
                    net_g.load_state_dict(
                        torch.load(hps.pretrainG, map_location="cpu")["model"]
                    )
                )
        if hps.pretrainD != "":
            if rank == 0:
                logger.info(f"Loaded pretrained discriminator from {hps.pretrainD}")
            if hasattr(net_d, "module"):
                logger.info(
                    net_d.module.load_state_dict(
                        torch.load(hps.pretrainD, map_location="cpu")["model"]
                    )
                )
            else:
                logger.info(
                    net_d.load_state_dict(
                        torch.load(hps.pretrainD, map_location="cpu")["model"]
                    )
                )

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )

    scaler = GradScaler(enabled=hps.train.fp16_run)

    cache = []
    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank == 0:
            train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d],
                [optim_g, optim_d],
                [scheduler_g, scheduler_d],
                train_loader,
                logger,
                writer,
                writer_eval,
                scaler,
            )
        else:
            train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d],
                [optim_g, optim_d],
                [scheduler_g, scheduler_d],
                train_loader,
                logger,
                None,
                None,
                scaler,
            )


def train_and_evaluate(
    rank,
    epoch,
    hps,
    nets,
    optimizers,
    schedulers,
    loader,
    logger,
    writer,
    writer_eval,
    scaler,
):
    net_g, net_d = nets
    optim_g, optim_d = optimizers
    scheduler_g, scheduler_d = schedulers
    epoch_recorder = EpochRecorder()

    if rank == 0:
        logger.info(f"Epoch {epoch} starting")
    for i, batch in enumerate(loader):
        if rank == 0:
            logger.info(f"Processing batch {i} of epoch {epoch}")

        # Forward pass
        with autocast():
            mel = batch["mel"].to("cuda")
            spec = batch["spec"].to("cuda")
            audio = batch["audio"].to("cuda")
            mel_target = batch["mel_target"].to("cuda")

            # Discriminator forward pass
            net_d_out = net_d(mel_target, audio)

            # Generator forward pass
            net_g_out = net_g(mel)

        # Calculate losses
        loss_d = discriminator_loss(net_d_out, mel_target)
        loss_g = generator_loss(net_g_out, spec)

        # Backward pass
        optimizer_d.zero_grad()
        scaler.scale(loss_d).backward()
        scaler.step(optimizer_d)

        optimizer_g.zero_grad()
        scaler.scale(loss_g).backward()
        scaler.step(optimizer_g)
        scaler.update()

        # Logging
        if rank == 0:
            logger.info(f"Epoch: {epoch}, Batch {i}: Discriminator loss: {loss_d.item()}, Generator loss: {loss_g.item()}")

    if rank == 0:
        writer.add_scalar("Discriminator Loss", loss_d.item(), epoch)
        writer.add_scalar("Generator Loss", loss_g.item(), epoch)
        logger.info(f"Epoch: {epoch} completed in {epoch_recorder.record()}")

    # Save checkpoints
    if epoch % hps.train.checkpoint_interval == 0:
        utils.save_checkpoint(
            os.path.join(hps.model_dir, f"G_{epoch:04d}.pth"),
            net_g,
            optim_g,
        )
        utils.save_checkpoint(
            os.path.join(hps.model_dir, f"D_{epoch:04d}.pth"),
            net_d,
            optim_d,
        )
        if rank == 0:
            logger.info(f"Checkpoint saved for epoch {epoch}")


if __name__ == "__main__":
    main()
