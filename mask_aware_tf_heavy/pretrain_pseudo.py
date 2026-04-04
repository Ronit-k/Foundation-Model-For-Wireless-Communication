# =============================================================================
# pretrain_pseudo.py — Pre-training for MAT Pseudo-Grid LWM (Method 2)
#
# Pre-trains the MAT Pseudo-Grid LWM hybrid model using masked channel
# modelling (MCM) on DeepMIMO wireless channel data.
#
# Mirrors the structure of lwm1_1_ca/pretraining.py:
#   • Same DeepMIMO scenarios and data loading
#   • Same hyperparameters (lr, batch_size, epochs, scheduler…)
#   • MCM MSE loss on 16-length masked patches
#
# Usage (from project root):
#   python -m mask_aware_tf.pretrain_pseudo [options]
#
# Quick start:
#   python -m mask_aware_tf.pretrain_pseudo \
#       --epochs 100 \
#       --batch-size 64 \
#       --save-path mask_aware_tf/mat_pseudo_lwm_weights.pth
# =============================================================================

import argparse
import os
import sys
import time

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

try:
    from torch import amp as torch_amp
except Exception:
    torch_amp = None

from lwm.input_preprocess import DeepMIMO_data_gen, deepmimo_data_cleaning
from mask_aware_tf.mat_pseudo_lwm import MATPseudoLWM


# =============================================================================
# Utilities
# =============================================================================

def default_num_workers() -> int:
    cpu_count = os.cpu_count() or 2
    return max(1, min(8, cpu_count // 2))


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-training for MAT Pseudo-Grid LWM (Method 2: 8×16)."
    )

    parser.add_argument("--scenarios", nargs="+", default=[
        "O1_3p5_v1", "O1_3p5_v2", "Boston5G_3p5", "asu_campus1",
    ])
    parser.add_argument("--dataset-folder", type=str, default=None)
    parser.add_argument("--channels-cache", type=str, default=None)

    parser.add_argument("--epochs",       type=int,   default=100)
    parser.add_argument("--batch-size",   type=int,   default=64)
    parser.add_argument("--train-ratio",  type=float, default=0.8)
    parser.add_argument("--val-ratio",    type=float, default=0.2)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--step-size",    type=int,   default=10)
    parser.add_argument("--gamma",        type=float, default=0.9)
    parser.add_argument("--seed",         type=int,   default=0)
    parser.add_argument("--snr-db",       type=float, default=None)
    parser.add_argument("--mask-ratio",   type=float, default=0.15)

    parser.add_argument("--device",              type=str, default=None)
    parser.add_argument("--num-workers",         type=int, default=default_num_workers())
    parser.add_argument("--prefetch-factor",     type=int, default=2)
    parser.add_argument("--persistent-workers",  action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pin-memory",          action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp",                 action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tf32",                action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cudnn-benchmark",     action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--torch-compile",       action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compile-mode",        type=str, default="default",
                        choices=["default", "reduce-overhead", "max-autotune"])

    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--log-file",     type=str, default=None)
    parser.add_argument("--tensorboard",  action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tb-logdir",    type=str, default="runs/mat_pseudo_lwm_pretraining")

    parser.add_argument("--scheduler-step-per-batch",
                        action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--save-path",  type=str, default="mask_aware_tf/mat_pseudo_lwm_weights.pth")
    parser.add_argument("--save-every", type=int, default=0)

    return parser.parse_args()


# =============================================================================
# Data Loading (same as lwm1_1_ca/pretraining.py)
# =============================================================================

def load_channels_ri(scenarios, dataset_folder=None, cache_path=None):
    if cache_path:
        expanded = os.path.expanduser(cache_path)
        for path in [expanded, expanded + ".npy"]:
            if os.path.exists(path):
                print(f"[Data] Loading cached channels from: {path}")
                channels_ri = np.load(path)
                # Squeeze any extra singleton dims to ensure (N, 2, H, W)
                while channels_ri.ndim > 4:
                    channels_ri = np.squeeze(channels_ri, axis=2)
                return channels_ri

    print(f"[Data] Generating channels for {len(scenarios)} scenarios …")
    data_parts = []
    for name in scenarios:
        print(f"  → {name}")
        kwargs = {}
        if dataset_folder is not None:
            kwargs["dataset_folder"] = dataset_folder
        deepmimo_data = DeepMIMO_data_gen(name, **kwargs)
        cleaned = deepmimo_data_cleaning(deepmimo_data)
        data_parts.append(cleaned)

    channels = np.concatenate(data_parts, axis=0)
    real = channels.real.astype(np.float32)
    imag = channels.imag.astype(np.float32)
    channels_ri = np.concatenate([real, imag], axis=1)

    if cache_path:
        save_path = os.path.expanduser(cache_path)
        cache_dir = os.path.dirname(save_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        np.save(save_path, channels_ri)
        print(f"[Data] Cached channels saved to: {save_path}.npy")

    return channels_ri


def split_data(dataset, train_ratio, val_ratio, seed=0):
    n = len(dataset)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)
    n_test = n - n_train - n_val
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.random_split(
        dataset, [n_train, n_val, n_test], generator=generator
    )


# =============================================================================
# Training Loop
# =============================================================================

def train_epoch(model, dataloader, optimizer, scheduler=None,
                device="cuda", amp=False, scaler=None, log_interval=0,
                log_fn=print, writer=None, epoch_idx=0, non_blocking=False,
                scheduler_step_per_batch=False):
    model.train()
    running_loss = 0.0
    data_time_sum = step_time_sum = samples_sum = 0.0
    log_loss = log_data_time = log_step_time = log_samples = 0.0
    end = time.perf_counter()

    for step, (channels,) in enumerate(dataloader):
        data_t = time.perf_counter() - end
        data_time_sum += data_t
        log_data_time += data_t

        channels = channels.to(device, non_blocking=non_blocking)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)

        if torch_amp is not None:
            autocast_ctx = torch_amp.autocast(device_type="cuda", enabled=amp)
        else:
            autocast_ctx = torch.cuda.amp.autocast(enabled=amp)

        with autocast_ctx:
            loss, _, _ = model(channels)

        if amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if scheduler is not None and scheduler_step_per_batch:
            scheduler.step()

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        step_t = time.perf_counter() - start
        step_time_sum += step_t
        log_step_time += step_t

        batch_sz = channels.size(0)
        samples_sum += batch_sz
        log_samples += batch_sz
        running_loss += loss.item()
        log_loss += loss.item()

        if log_interval and (step + 1) % log_interval == 0:
            avg_l = log_loss / log_interval
            avg_d = log_data_time / log_interval
            avg_s = log_step_time / log_interval
            throughput = log_samples / log_step_time if log_step_time > 0 else 0.0

            if writer is not None:
                g = epoch_idx * len(dataloader) + step + 1
                writer.add_scalar("train/loss_step", avg_l, g)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], g)

            log_fn(
                f"  step {step + 1:>5}/{len(dataloader)} | "
                f"loss {avg_l:.4f} | "
                f"data {avg_d * 1000:.1f} ms | "
                f"step {avg_s * 1000:.1f} ms | "
                f"{throughput:.1f} samples/s"
            )
            log_loss = log_data_time = log_step_time = log_samples = 0.0

        end = time.perf_counter()

    avg_loss = running_loss / len(dataloader)
    metrics = {
        "data_time":  data_time_sum / len(dataloader),
        "step_time":  step_time_sum / len(dataloader),
        "throughput": samples_sum / step_time_sum if step_time_sum > 0 else 0.0,
    }
    return avg_loss, metrics


def validate_epoch(model, dataloader, device="cuda", amp=False, non_blocking=False):
    if dataloader is None or len(dataloader) == 0:
        return 0.0, {"data_time": 0.0, "step_time": 0.0, "throughput": 0.0}

    model.eval()
    running_loss = 0.0
    data_time_sum = step_time_sum = samples_sum = 0.0
    end = time.perf_counter()

    with torch.inference_mode():
        for (channels,) in dataloader:
            data_t = time.perf_counter() - end
            data_time_sum += data_t

            channels = channels.to(device, non_blocking=non_blocking)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            start = time.perf_counter()

            if torch_amp is not None:
                autocast_ctx = torch_amp.autocast(device_type="cuda", enabled=amp)
            else:
                autocast_ctx = torch.cuda.amp.autocast(enabled=amp)

            with autocast_ctx:
                loss, _, _ = model(channels)

            if device.startswith("cuda"):
                torch.cuda.synchronize()
            step_t = time.perf_counter() - start
            step_time_sum += step_t
            samples_sum += channels.size(0)
            running_loss += loss.item()
            end = time.perf_counter()

    avg_loss = running_loss / len(dataloader)
    metrics = {
        "data_time":  data_time_sum / len(dataloader),
        "step_time":  step_time_sum / len(dataloader),
        "throughput": samples_sum / step_time_sum if step_time_sum > 0 else 0.0,
    }
    return avg_loss, metrics


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = args.cudnn_benchmark
        torch.backends.cuda.matmul.allow_tf32 = args.tf32
        torch.backends.cudnn.allow_tf32 = args.tf32
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high" if args.tf32 else "highest")

    channels_ri = load_channels_ri(
        scenarios=args.scenarios,
        dataset_folder=args.dataset_folder,
        cache_path=args.channels_cache,
    )
    print(f"[Data] Channels shape: {channels_ri.shape} (N, 2, H, W)")

    tensor = torch.from_numpy(channels_ri)
    dataset = torch.utils.data.TensorDataset(tensor)
    train_data, val_data, test_data = split_data(
        dataset, args.train_ratio, args.val_ratio, seed=args.seed
    )

    num_workers = max(0, int(args.num_workers))
    pin_memory = bool(args.pin_memory and device.startswith("cuda"))
    non_blocking = pin_memory and device.startswith("cuda")

    loader_kwargs = {
        "batch_size": args.batch_size,
        "pin_memory": pin_memory,
        "num_workers": num_workers,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"]    = args.prefetch_factor
        loader_kwargs["persistent_workers"] = args.persistent_workers

    train_loader = torch.utils.data.DataLoader(train_data, shuffle=True,  **loader_kwargs)
    val_loader   = torch.utils.data.DataLoader(val_data,   shuffle=False, **loader_kwargs) if len(val_data) > 0 else None
    test_loader  = torch.utils.data.DataLoader(test_data,  shuffle=False, **loader_kwargs) if len(test_data) > 0 else None

    model = MATPseudoLWM(
        gen_raw=False, snr_db=args.snr_db, mask_ratio=args.mask_ratio,
    ).to(device)
    print(f"[Model] MATPseudoLWM created. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    if args.torch_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode=args.compile_mode)
            print(f"[Model] torch.compile enabled (mode={args.compile_mode})")
        except Exception as exc:
            print(f"[Warning] torch.compile failed: {exc}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    amp_enabled = bool(args.amp and device.startswith("cuda"))
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    log_file = args.log_file
    if log_file is None:
        base = args.save_path or "mask_aware_tf/pretrain_pseudo"
        log_file = os.path.splitext(base)[0] + ".log"
    log_file = os.path.expanduser(log_file)
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    log_fp = open(log_file, "a", encoding="utf-8")

    writer = None
    if args.tensorboard:
        if SummaryWriter is None:
            print("[Warning] TensorBoard not installed.")
        else:
            tb_dir = os.path.expanduser(args.tb_logdir)
            os.makedirs(tb_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=tb_dir)

    def log(*msg):
        print(*msg)
        log_fp.write(" ".join(str(m) for m in msg) + "\n")
        log_fp.flush()

    log("=" * 60)
    log("MAT Pseudo-Grid LWM (Method 2: 8×16) — Pre-training")
    log("=" * 60)
    log(f"Log file    : {log_file}")
    log(f"Device      : {device}")
    log(f"Scenarios   : {args.scenarios}")
    log(f"AMP         : {'enabled' if amp_enabled else 'disabled'}")
    log(f"Epochs      : {args.epochs}")
    log(f"Batch size  : {args.batch_size}")
    log(f"LR          : {args.lr} | Weight decay: {args.weight_decay}")
    log(f"Scheduler   : StepLR(step={args.step_size}, gamma={args.gamma})")
    log(f"Mask ratio  : {args.mask_ratio}")
    log(f"Save path   : {args.save_path}")
    log("-" * 60)
    log(f"Dataset — train: {len(train_data)}, val: {len(val_data)}, test: {len(test_data)}")
    log("=" * 60)

    for epoch in range(args.epochs):
        log(f"\nEpoch {epoch + 1}/{args.epochs}")
        log("-" * 40)
        log(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()

        train_loss, train_m = train_epoch(
            model, train_loader, optimizer, scheduler,
            device=device, amp=amp_enabled, scaler=scaler,
            log_interval=args.log_interval, log_fn=log,
            writer=writer, epoch_idx=epoch, non_blocking=non_blocking,
            scheduler_step_per_batch=args.scheduler_step_per_batch,
        )

        if scheduler is not None and not args.scheduler_step_per_batch:
            scheduler.step()

        log(f"Training Loss: {train_loss:.6f} | "
            f"{train_m['throughput']:.1f} samples/s")

        if writer is not None:
            writer.add_scalar("train/loss_epoch", train_loss, epoch + 1)

        if device.startswith("cuda"):
            peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
            log(f"Peak GPU memory: {peak:.1f} MB")

        if val_loader is not None:
            val_loss, val_m = validate_epoch(
                model, val_loader, device=device, amp=amp_enabled,
                non_blocking=non_blocking,
            )
            log(f"Validation Loss: {val_loss:.6f} | {val_m['throughput']:.1f} samples/s")
            if writer is not None:
                writer.add_scalar("val/loss_epoch", val_loss, epoch + 1)

        if args.save_every and (epoch + 1) % args.save_every == 0 and args.save_path:
            ckpt_dir = os.path.dirname(args.save_path)
            if ckpt_dir:
                os.makedirs(ckpt_dir, exist_ok=True)
            save_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            ckpt = os.path.splitext(args.save_path)[0] + f"_epoch{epoch + 1}.pth"
            torch.save(save_model.state_dict(), ckpt)
            log(f"Checkpoint saved: {ckpt}")

    if test_loader is not None:
        test_loss, test_m = validate_epoch(
            model, test_loader, device=device, amp=amp_enabled,
            non_blocking=non_blocking,
        )
        log(f"\nTest Loss: {test_loss:.6f} | {test_m['throughput']:.1f} samples/s")

    if args.save_path:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        save_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        torch.save(save_model.state_dict(), args.save_path)
        log(f"\nFinal model saved to: {args.save_path}")

    if writer is not None:
        writer.close()
    log_fp.close()
    print("Pre-training complete.")


if __name__ == "__main__":
    main()
