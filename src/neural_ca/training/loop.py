"""Training loop for the Growing NCA (see docs/DESIGN.md §5).

Recipe: sample a batch of grid states from the pool, reset the worst to the seed (so the model
keeps learning to grow from scratch) and damage the best (so it learns to regenerate), run the
NCA for a random number of steps, and match the target with MSE. Per-parameter gradient
normalisation keeps the notoriously spiky NCA gradients stable.
"""

from pathlib import Path

import torch
import torch.nn.functional as F

from neural_ca.config import Config
from neural_ca.data.state import make_seed_state
from neural_ca.data.targets import load_target, make_target, pad_to_grid
from neural_ca.model.nca import NCA
from neural_ca.training.pool import SamplePool, damage
from neural_ca.utils import resolve_device, set_seed


def build_target(cfg: Config) -> torch.Tensor:
    """Build the padded RGBA target from the config (procedural name or image path)."""
    name = cfg.data.target
    target = (
        load_target(name, cfg.data.size) if Path(name).suffix else make_target(name, cfg.data.size)
    )
    return pad_to_grid(target, cfg.data.pad)


def train(
    cfg: Config,
    device: torch.device | None = None,
    log_every: int = 100,
    log_dir: str | Path | None = None,
) -> dict:
    """Train an NCA and return the model, loss history, and target.

    If ``log_dir`` is set, per-step losses are also written there for TensorBoard.
    """
    set_seed(cfg.seed)
    device = device or resolve_device(cfg.device)

    writer = None
    if log_dir is not None:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(str(log_dir))

    target = build_target(cfg).to(device)  # (4, H, W)
    grid = target.shape[-1]
    targets = target.unsqueeze(0)  # broadcast over the batch

    model = NCA(
        cfg.model.channels, cfg.model.hidden, cfg.model.fire_rate, cfg.model.alive_threshold
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    # Drop the LR late in training to sharpen the final pattern (docs/DESIGN.md §5).
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[max(1, int(cfg.train.steps * 0.7))], gamma=0.3
    )

    seed_state = make_seed_state(1, cfg.model.channels, grid)[0].to(device)  # (C, H, W)
    pool = SamplePool(seed_state.cpu(), cfg.train.pool_size)
    use_amp = device.type == "cuda" and cfg.train.precision == "bf16"
    lo, hi = cfg.train.step_range

    history: list[tuple[int, float]] = []
    for it in range(1, cfg.train.steps + 1):
        idx, batch = pool.sample(cfg.train.batch_size)
        batch = batch.to(device)

        # Rank by current fidelity so we can reset the worst and damage the best.
        with torch.no_grad():
            per_sample = F.mse_loss(
                batch[:, :4], targets.expand_as(batch[:, :4]), reduction="none"
            ).mean(dim=(1, 2, 3))
        order = torch.argsort(per_sample, descending=True)
        batch, idx = batch[order], idx[order.cpu()]
        batch[0] = seed_state  # worst → fresh seed (anchors "grow from scratch")
        if cfg.train.damage and batch.shape[0] > 2:
            batch[-2:] = damage(batch[-2:])  # best few → damaged (learn to regenerate)

        steps = int(torch.randint(lo, hi + 1, (1,)).item())
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            out = model(batch, steps)
            loss = F.mse_loss(out[:, :4], targets.expand_as(out[:, :4]))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        for p in model.parameters():  # per-parameter gradient normalisation (stability)
            if p.grad is not None:
                p.grad /= p.grad.norm() + 1e-8
        opt.step()
        sched.step()

        pool.commit(idx, out)
        if it == 1 or it % log_every == 0:
            value = loss.detach().item()
            history.append((it, value))
            if writer is not None:
                writer.add_scalar("loss/mse", value, it)

    if writer is not None:
        writer.close()
    return {"model": model, "history": history, "target": target}
