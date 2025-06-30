import os
# Makes SSIM deterministic on GPU
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# Prefents memory secmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import random
import numpy as np
import gc
from tqdm import tqdm
import torch
import torch.distributed as dist
import torch.optim as optim
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, DataLoader, Subset
from Loss_Function import edge_loss, loss_funct
from TrainDataset import SpkDataset
from ..model.DepthModel import BEPDepthNetwork
from ..model.DepthModel import FixedSobel
import utils
import torch.backends.cudnn as cudnn

cudnn.deterministic = True
cudnn.benchmark = False


import warnings
warnings.filterwarnings("ignore", message=".*torch.cpu.amp.autocast.*")
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
def train_worker(local_rank: int, gpu_ids: list, data_dir: str, model_dir: str, batch_size: int = 4, epochs: int = 180,
    split_ratio: tuple = (0.8, 0.10, 0.10)):
    """
    DDP worker entrypoint. Spawned once per GPU.

    Args:
        local_rank: index of this process (0..world_size-1)
        gpu_ids: list of GPU device IDs to use
        data_dir: path to dataset
        model_dir: path to save models/logs
        batch_size: per-GPU batch size
        epochs: number of training epochs
        split_ratio: train/val/test split ratios
    """
    # Initialize process group
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    world_size = len(gpu_ids)
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=local_rank
    )

    # Set device for this process
    gpu = gpu_ids[local_rank]
    torch.cuda.set_device(gpu)
    device = torch.device(f'cuda:{gpu}')

    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # Build model and wrap in DDP
    model = BEPDepthNetwork()
    model.to(device)
    model = DDP(
        model,
        device_ids=[gpu],
        output_device=gpu,
        find_unused_parameters=False
    )

    # Prepare dataset and samplers
    dataset = SpkDataset(data_dir, device=None)
    train_idx, val_idx, test_idx = dataset.split_dataset(split_ratio)
    if local_rank == 0:
        # only rank 0 writes the split file
        os.makedirs(model_dir, exist_ok=True)
        dataset.save_splits(
            os.path.join(model_dir, "dataset_split.txt"),
            train_idx, val_idx, test_idx
        )
    g = torch.Generator()
    g.manual_seed(42)
    train_sampler = DistributedSampler(
        Subset(dataset, train_idx),
        num_replicas=world_size,
        rank=local_rank,
        seed=42,
        shuffle=True
    )
    val_sampler = DistributedSampler(
        Subset(dataset, val_idx),
        num_replicas=world_size,
        rank=local_rank,
        seed=42,
        shuffle=False
    )
    test_sampler = DistributedSampler(
        Subset(dataset, test_idx),
        num_replicas=world_size,
        rank=local_rank,
        seed=42,
        shuffle=False
    )
    train_loader = DataLoader(
        train_sampler.dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=seed_worker
    )
    val_loader = DataLoader(
        val_sampler.dataset,
        batch_size=batch_size*3,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=seed_worker
    )
    test_loader = DataLoader(
        test_sampler.dataset,
        batch_size=batch_size*3,
        sampler=test_sampler,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=seed_worker
    )

    # Loss, optimizer, scaler

    steps_per_epoch = len(train_loader)
    l1_loss = nn.L1Loss().to(device)
    sobel = FixedSobel().to(device)
    base_lr = 3e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, betas=(0.9, 0.999), weight_decay= 5e-4)
    # schedule: 5-epoch linear warm-up + cosine decay


    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=3e-6)

    scaler = torch.amp.GradScaler(init_scale=1024, growth_interval=200)

    # 7) Training loop
    for epoch in range(epochs):
        iterator = (
            tqdm(train_loader, total=len(train_loader),
                    desc="Training: Epoch={ep}, Step".format(ep=epoch))
            if local_rank == 0 else train_loader
        )

        train_sampler.set_epoch(epoch)
        model.train()
        stats = {"depth_loss": [], "lr": [], "param_norm": [], "grad_norm": []}

        for batch in iterator:
            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                inputs = batch["input_spk"].to(device, dtype=torch.float16)
                labels = batch["depth_off"].to(device)
                depth_map, edge, disp1_8, disp1_16, disp1_32 = model(inputs)

                edge_label = sobel(labels.unsqueeze(1))

            loss_edge = edge_loss(edge, edge_label)
            loss1 = loss_funct(depth_map, labels, epoch)
            loss2 = loss_funct(disp1_8, labels, epoch)
            loss3 = loss_funct(disp1_16, labels, epoch)
            loss4 = loss_funct(disp1_32, labels, epoch)

            aux_w  = 1.0 if epoch < 30 else 0.05  # Gives auxillery loss loss more weight in early epochs.
            edge_w = min(0.5, epoch / 80.)   # Gives edge loss more weight in later epochs and less at the start
            loss = 1.0 * loss1 + aux_w * (0.5 * loss2 + 0.25 * loss3 + 0.125 * loss4) + edge_w * loss_edge
            scaler.scale(loss).backward()
            # scale grads back to Fp32 for saving (happens normally in scalar.step())
            scaler.unscale_(optimizer)
            # save stats.
            with torch.no_grad():
                # total ‖θ‖₂
                param_sqsum = torch.sum(torch.stack([
                    p.pow(2).sum() for p in model.parameters()
                ]))
                param_norm = torch.sqrt(param_sqsum)

                # (skip params that did not receive a grad)
                grad_sqsum = torch.sum(torch.stack([
                    p.grad.pow(2).sum()
                    for p in model.parameters() if p.grad is not None
                ]))
                grad_norm = torch.sqrt(grad_sqsum)

            # Gradient clipping
            max_norm = 1
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            stats["param_norm"].append(param_norm.item())
            stats["grad_norm"].append(grad_norm.item())
            scaler.step(optimizer)
            scaler.update()
            stats["lr"].append(scheduler.get_last_lr()[0])
            stats["depth_loss"].append(loss.item())
        scheduler.step()


        # Save training logs & checkpoint on rank 0
        all_stats = [None] * world_size
        dist.all_gather_object(all_stats, stats)
        if local_rank == 0:
            merged = {"depth_loss": [], "lr": [], "param_norm": [], "grad_norm": []}
            for r in range(world_size):
                for k in merged:
                    merged[k].extend(all_stats[r][k])
            utils.save_training_history(
                merged,
                f"depth_train_stats_{epoch}",
                os.path.join(model_dir, "training_logs")
            )

            if epoch % 10 == 0 or epoch == epochs - 1:
                torch.save(model.module.state_dict(), os.path.join(model_dir, f"bep1c_epoch_{epoch}.pth"))
        dist.barrier()
        torch.cuda.empty_cache()
        gc.collect()
        # Validation
        model.eval()
        stats = {"depth_loss": []}
        with torch.no_grad():
            iterator = (
                tqdm(val_loader, total=len(val_loader),
                     desc="Validation: Epoch={ep}, Step".format(ep=epoch))
                if local_rank == 0 else val_loader
            )
            val_sampler.set_epoch(epoch)
            for batch in iterator:
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    inputs = batch["input_spk"].to(device, dtype=torch.float16)
                    labels = batch["depth_off"].to(device)
                    outputs, _ = model(inputs)
                    loss = l1_loss(outputs.squeeze(1), labels)
                stats["depth_loss"].append(loss.item())

        all_losses = [None] * world_size
        dist.all_gather_object(all_losses, stats["depth_loss"])
        if local_rank == 0:
            # flatten the list of lists
            flat = [l for sub in all_losses for l in sub]
            utils.save_training_history(
                {"depth_loss": flat},
                f"depth_val_stats_{epoch}",
                os.path.join(model_dir, "training_logs")
            )
        dist.barrier()

    # Testing
    model.eval()
    stats = {"depth_loss": []}
    iterator = (
        tqdm(test_loader, total=len(test_loader),
             desc="Testing, Step")
        if local_rank == 0 else test_loader
    )
    with torch.no_grad():
        for batch in iterator:
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                inputs = batch["input_spk"].to(device, dtype=torch.float16)
                labels = batch["depth_off"].to(device)
                outputs, _ = model(inputs)
                loss = l1_loss(outputs.squeeze(1), labels)
            stats["depth_loss"].append(loss.item())

    all_losses = [None] * world_size
    dist.all_gather_object(all_losses, stats["depth_loss"])
    if local_rank == 0:
        # flatten the list of lists
        flat = [l for sub in all_losses for l in sub]
        utils.save_training_history(
            {"depth_loss": flat},
            f"test_stats",
            os.path.join(model_dir, "training_logs")
        )
    dist.barrier()

    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    import argparse, os, torch.multiprocessing as mp

    parser = argparse.ArgumentParser(
        description="Train SNN on provided dataset directory with DDP."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the dataset directory."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Where to save model checkpoints."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size per GPU (defaults to 4)."
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        required=True,
        help="Comma-separated list of GPU IDs to use (e.g. \"0,1,4\")."
    )

    args = parser.parse_args()
    os.makedirs(args.model_dir, exist_ok=True)

    # Parse comma‑separated GPU list → [int, ...]
    gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip().isdigit()]

    # Spawn one process per GPU
    mp.spawn(
        train_worker,
        args=(
            gpu_ids,
            args.data_dir,
            args.model_dir,
            args.batch_size,
        ),
        nprocs=len(gpu_ids),
        join=True
    )