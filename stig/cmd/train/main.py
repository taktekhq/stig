"""
Command train trains a model on a given dataset.
"""

import copy
from pathlib import Path
from typing import Tuple

import tqdm, torch, torch.nn as nn, torch.utils.data as data

from stig.internal.env import env
from stig.internal.dataset.dataset import get_or_create_dataset
from stig.internal.model.model import StigNet


def _class_weights(labels: torch.Tensor, num_classes: int = 3) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=num_classes).float()
    present = counts > 0
    weights = torch.ones(num_classes, dtype=torch.float32)
    if present.any():
        total = counts.sum()
        inv = torch.sqrt(total / (counts.clamp(min=1.0) * present.sum().float()))
        weights = torch.where(present, inv.clamp(0.5, 3.0), torch.ones_like(inv))
        weights = weights / weights[present].mean()
    return weights


def _augment_batch(
    x: torch.Tensor,
    s: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply horizontal flip (swapping left<->right steering) and mild
    brightness/contrast jitter on a uint8 batch (B, 3, H, W).
    """
    x = x.clone()
    s = s.clone()

    # Horizontal flip with p=0.5: swap steering 1 (left) <-> 2 (right).
    flip_mask = torch.rand(x.size(0), device=x.device) < 0.5
    if flip_mask.any():
        x[flip_mask] = torch.flip(x[flip_mask], dims=[-1])
        s_flipped = s[flip_mask]
        is_left = s_flipped == 1
        is_right = s_flipped == 2
        s_flipped = torch.where(is_left, torch.tensor(2, device=s.device), s_flipped)
        s_flipped = torch.where(is_right, torch.tensor(1, device=s.device), s_flipped)
        s[flip_mask] = s_flipped

    # Mild brightness/contrast jitter.
    scale = torch.empty((x.size(0), 1, 1, 1), device=x.device).uniform_(0.88, 1.12)
    shift = torch.empty((x.size(0), 1, 1, 1), device=x.device).uniform_(-10.0, 10.0)
    x = (x.float() * scale + shift).clamp(0.0, 255.0).to(torch.uint8)

    return x, s


def train(
    model_name: str,
    datasets_dir: str,
    models_dir: str,
    recordings_dir: str,
    frame_size: Tuple[int, int],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
):
    dataset = get_or_create_dataset(model_name, datasets_dir, recordings_dir, frame_size)

    # Deterministic 90/10 train/val split.
    n_total = len(dataset)
    n_val = max(1, int(n_total * 0.1))
    n_train = n_total - n_val
    gen = torch.Generator().manual_seed(42)
    train_ds, val_ds = data.random_split(dataset, [n_train, n_val], generator=gen)

    train_loader = data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Compute class weights from the training split.
    train_indices = torch.tensor(train_ds.indices, dtype=torch.long)
    all_t = dataset.tensors[1][train_indices]
    all_s = dataset.tensors[2][train_indices]
    w_t = _class_weights(all_t, 3).to(device)
    w_s = _class_weights(all_s, 3).to(device)
    # Note: horizontal flip balances left (1) and right (2), so equalize their weights.
    lr_w = 0.5 * (w_s[1] + w_s[2])
    w_s[1] = lr_w
    w_s[2] = lr_w

    print(
        f"Dataset: total={n_total} train={n_train} val={n_val} "
        f"t_counts={torch.bincount(all_t, minlength=3).tolist()} "
        f"s_counts={torch.bincount(all_s, minlength=3).tolist()}"
    )

    net = StigNet(frame_size).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    ce_t = nn.CrossEntropyLoss(weight=w_t)
    ce_s = nn.CrossEntropyLoss(weight=w_s)

    best_val_loss = float("inf")
    best_state = copy.deepcopy(net.state_dict())

    for ep in range(epochs):
        net.train()
        tot = tc = sc = 0
        tr_loss = 0.0

        for x, t, s in tqdm.tqdm(train_loader, desc=f"epoch {ep:02d}"):
            x = x.to(device)
            t = t.to(device)
            s = s.to(device)
            x, s = _augment_batch(x, s)

            ot, os = net(x)
            loss = ce_t(ot, t) + 1.5 * ce_s(os, s)

            opt.zero_grad()
            loss.backward()
            opt.step()

            bs = x.size(0)
            tot += bs
            tr_loss += loss.item() * bs
            tc += (ot.argmax(1) == t).sum().item()
            sc += (os.argmax(1) == s).sum().item()

        sched.step()

        net.eval()
        vtot = vtc = vsc = 0
        val_loss = 0.0
        with torch.inference_mode():
            for x, t, s in val_loader:
                x = x.to(device)
                t = t.to(device)
                s = s.to(device)
                ot, os = net(x)
                loss = ce_t(ot, t) + 1.5 * ce_s(os, s)
                bs = x.size(0)
                vtot += bs
                val_loss += loss.item() * bs
                vtc += (ot.argmax(1) == t).sum().item()
                vsc += (os.argmax(1) == s).sum().item()

        val_loss /= max(1, vtot)
        tr_loss /= max(1, tot)
        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(net.state_dict())
            marker = " *"

        print(
            f"epoch {ep:02d}\t"
            f"tr_loss={tr_loss:.3f}\tval_loss={val_loss:.3f}\t"
            f"t_acc={tc/tot:.3f}/{vtc/vtot:.3f}\t"
            f"s_acc={sc/tot:.3f}/{vsc/vtot:.3f}{marker}"
        )

    net.load_state_dict(best_state)
    net.eval()

    Path(models_dir).mkdir(parents=True, exist_ok=True)
    model_path = Path(models_dir) / f"{model_name}.pt"
    torch.jit.script(net.cpu()).save(str(model_path))
    print(f"✅ saved {model_path} (best val_loss={best_val_loss:.3f})")


if __name__ == "__main__":
    model_name = env.lookup("MODEL_NAME")

    datasets_dir = env.lookup("DATASETS_DIR")
    models_dir = env.lookup("MODELS_DIR")
    recordings_dir = env.lookup("RECORDINGS_DIR")

    frame_height = env.lookupInt("FRAME_HEIGHT")
    frame_width = env.lookupInt("FRAME_WIDTH")
    frame_size = (frame_height, frame_width)

    epochs = env.lookupInt("EPOCHS")
    batch_size = env.lookupInt("BATCH_SIZE")
    learning_rate = env.lookupFloat("LEARNING_RATE")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train(
        model_name,
        datasets_dir,
        models_dir,
        recordings_dir,
        frame_size,
        epochs,
        batch_size,
        learning_rate,
        device,
    )
