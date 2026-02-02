import os, re
import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from experiments.heat_map_att_kernel3x3 import *
from src.experiments.heat_map_att_grid import *




def _softmax_local(attn_logits, k2=9):
    B, C, H, W = attn_logits.shape
    assert C % k2 == 0, f"Esperaba C múltiplo de {k2}. Got C={C}."
    heads = C // k2
    w = attn_logits.view(B, heads, k2, H, W)
    w = torch.softmax(w, dim=2)
    return w  # (B, heads, 9, H, W)


def _kernel_at(weights_5d, y, x):
    # weights_5d: (B, heads, 9, H, W)
    kern = weights_5d[:, :, :, y, x].mean(dim=1)
    return kern.view(-1, 3, 3)


def _get_random_batch(loader, device, seed=None):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    if hasattr(loader, "__len__"):
        n_batches = len(loader)
        j = random.randrange(n_batches)
        it = iter(loader)
        for _ in range(j):
            next(it)
        batch = next(it)
    else:
        batch = next(iter(loader))

    x = batch[0] if isinstance(batch, (tuple, list)) else batch
    return x.to(device)

def _choose_random_indices(B, n_images, seed=None):
    if seed is not None:
        random.seed(seed + 12345)
    n_images = min(n_images, B)
    return random.sample(range(B), k=n_images)

def sample_q_indices(Hg, Wg, n_q=32, seed=0, exclude_border=1, device="cpu"):
    """
    Uniform sample query indices in Hg×Wg token grid.
    exclude_border=1 removes the outer ring (reduces border artifacts).
    """
    ys = torch.arange(Hg, device=device)
    xs = torch.arange(Wg, device=device)
    Y, X = torch.meshgrid(ys, xs, indexing="ij")

    if exclude_border > 0:
        mask = (Y >= exclude_border) & (Y < Hg - exclude_border) & \
               (X >= exclude_border) & (X < Wg - exclude_border)
        valid = torch.nonzero(mask.reshape(-1), as_tuple=False).flatten()
    else:
        valid = torch.arange(Hg * Wg, device=device)

    rng = np.random.default_rng(seed)
    if len(valid) <= n_q:
        return valid.tolist()
    idx = rng.choice(len(valid), size=n_q, replace=False)
    return valid[idx].tolist()


def sample_xy(H, W, n_xy=64, seed=0, exclude_border=1):
    """
    Uniform sample spatial positions (y,x) in feature map H×W.
    exclude_border=1 ensures 3×3 neighborhood exists (for Outlooker kernels).
    """
    rng = np.random.default_rng(seed)
    ys = np.arange(exclude_border, H - exclude_border)
    xs = np.arange(exclude_border, W - exclude_border)
    Y, X = np.meshgrid(ys, xs, indexing="ij")
    coords = np.stack([Y.reshape(-1), X.reshape(-1)], axis=1)

    if len(coords) == 0:
        return []

    if len(coords) <= n_xy:
        return coords.tolist()
    idx = rng.choice(len(coords), size=n_xy, replace=False)
    return coords[idx].tolist()

def grid_attn_mad_for_query(attn, meta, Hg, Wg, g, b, gy, gx, q_idx, head_reduce="mean"):
    """
    attn: [Bgrp, heads, N, N]
    meta: (B, Hf, Wf, C, g)
    q_idx: query index in 0..N-1, N=Hg*Wg
    gy,gx: which interleaving group
    Returns MAD L1 in full featuremap coords (Hf,Wf).
    """
    B, Hf, Wf, C, g_meta = meta
    assert g_meta == g
    N = Hg * Wg

    grp = b * (g*g) + gy * g + gx
    A = attn[grp]  # [heads, N, N]

    if head_reduce == "mean":
        A = A.mean(0)        # [N,N]
    elif head_reduce == "max":
        A = A.max(0).values  # [N,N]
    else:
        raise ValueError("head_reduce must be 'mean' or 'max'")

    w = A[q_idx]                          # [N]
    w = w / (w.sum() + 1e-12)

    # query/key coords in grid-local coords
    qy = q_idx // Wg
    qx = q_idx %  Wg

    ky = torch.arange(Hg, device=w.device).repeat_interleave(Wg)
    kx = torch.arange(Wg, device=w.device).repeat(Hg)

    # map to full featuremap coords by interleaving
    yq_full = qy * g + gy
    xq_full = qx * g + gx
    yk_full = ky * g + gy
    xk_full = kx * g + gx

    dist_l1 = (yk_full - yq_full).abs() + (xk_full - xq_full).abs()  # [N]
    mad = (w * dist_l1).sum().item()
    return mad


def grid_attn_mad_summary(attn, meta, Hg, Wg, g, b=0, gy=0, gx=0, q_idxs=(None,)):
    """
    Average MAD over multiple queries q_idxs.
    """
    out = []
    for q in q_idxs:
        out.append(grid_attn_mad_for_query(attn, meta, Hg, Wg, g, b, gy, gx, q))
    return float(sum(out) / len(out))


def outlooker_kernel_mad_norm(k3x3: torch.Tensor, eps=1e-12):
    """
    k3x3: [3,3] non-negative weights (ideally sum to 1).
    Returns MAD normalized to [0,1] by dividing by 2.
    """
    k = torch.clamp(k3x3, min=0.0)
    k = k / (k.sum() + eps)
    dist = torch.tensor([[2,1,2],
                         [1,0,1],
                         [2,1,2]], device=k.device, dtype=k.dtype)
    mad = (k * dist).sum()         # in [0,2]
    return (mad / 2.0).item()


def outlooker_mad_for_image_sampled(attn_logits_b: torch.Tensor, n_xy=64, seed=0, exclude_border=1):
    w = _softmax_local(attn_logits_b, k2=9)
    H, W = int(w.shape[1]), int(w.shape[2])

    # --- adaptive border ---
    eb = int(exclude_border)
    if H - 2*eb <= 0 or W - 2*eb <= 0:
        eb = 0  # fallback: allow borders

    coords = sample_xy(H, W, n_xy=n_xy, seed=seed, exclude_border=eb)
    if len(coords) == 0:
        # last-resort fallback: sample from all pixels
        coords = sample_xy(H, W, n_xy=n_xy, seed=seed, exclude_border=0)
        if len(coords) == 0:
            return None, None

    mads = []
    for (y, x) in coords:
        k = _kernel_at(w, y, x)[0]   # [3,3]
        mads.append(outlooker_kernel_mad_norm(k))

    return float(np.mean(mads)), float(np.std(mads))

@torch.no_grad()
def compute_grid_and_outlooker_mad_by_stage(
    model,
    loader,
    block_idx=0,
    stages=(0,1,2,3),
    n_images=64,
    seed=10,
    device="cuda",
    normalize_grid=True,
    # NEW sampling controls
    grid_n_q=32,
    grid_exclude_border=1,
    grid_avg_over_groups=True,
    out_n_xy=64,
    out_exclude_border=1,
):
    model = model.to(device).eval()

    x_all = _get_random_batch(loader, device=device, seed=seed)
    B = x_all.shape[0]
    idxs = _choose_random_indices(B, n_images=n_images, seed=seed)
    x = x_all[idxs]
    n = x.shape[0]

    enable_mhsa_capture(model, True)
    cap_grid = GridAttnCapturer(model)
    cap_out  = OutlookAttnCapturer(model)

    _ = model(x)

    results = []

    for s in stages:
        # ------------------
        # GRID
        # ------------------
        pack_g = cap_grid.get(stage=s, block=block_idx)
        grid_ok = (pack_g is not None and pack_g.get("attn", None) is not None and pack_g.get("meta", None) is not None)

        grid_mean = grid_std = None
        grid_abs_mean = None
        Hf = Wf = None
        grid_denom = None

        if grid_ok:
            attn = pack_g["attn"]     # [Bgrp, heads, N, N]
            meta = pack_g["meta"]     # (B, Hf, Wf, C, g)
            Hg, Wg = pack_g["grid_hw"]
            g = pack_g["g"]

            Bm, Hf, Wf, C, _ = meta
            assert Bm == n

            grid_denom = float((Hf - 1) + (Wf - 1)) if normalize_grid else 1.0

            per_image = []
            for b in range(n):
                # average across all interleaving groups if requested
                group_vals = []
                if grid_avg_over_groups:
                    gy_range = range(g)
                    gx_range = range(g)
                else:
                    gy_range = [0]
                    gx_range = [0]

                for gy in gy_range:
                    for gx in gx_range:
                        q_seed = seed + 100000*s + 1000*b + 97*gy + 131*gx + 17*block_idx
                        q_idxs = sample_q_indices(
                            Hg, Wg,
                            n_q=grid_n_q,
                            seed=q_seed,
                            exclude_border=grid_exclude_border,
                            device=attn.device
                        )
                        if len(q_idxs) == 0:
                            continue
                        mad_abs = grid_attn_mad_summary(attn, meta, Hg, Wg, g, b=b, gy=gy, gx=gx, q_idxs=q_idxs)
                        mad = mad_abs / grid_denom if normalize_grid else mad_abs
                        group_vals.append(mad)

                if len(group_vals):
                    per_image.append(float(np.mean(group_vals)))

            if len(per_image):
                grid_mean = float(np.mean(per_image))
                grid_std  = float(np.std(per_image))
                grid_abs_mean = grid_mean * grid_denom if normalize_grid else grid_mean

        # ------------------
        # OUTLOOKER
        # ------------------
        attn_logits = cap_out.get(stage=s, block=block_idx)
        out_ok = (attn_logits is not None)

        out_mean = out_std = None
        out_abs_mean = None

        if out_ok:
            per_image = []
            for b in range(n):
                o_seed = seed + 200000*s + 1000*b + 19*block_idx
                mu, sd = outlooker_mad_for_image_sampled(
                    attn_logits[b:b+1],
                    n_xy=out_n_xy,
                    seed=o_seed,
                    exclude_border=out_exclude_border
                )
                if mu is not None:
                    per_image.append(mu)

            if len(per_image):
                out_mean = float(np.mean(per_image))  # already norm in [0,1]
                out_std  = float(np.std(per_image))
                out_abs_mean = out_mean * 2.0         # abs scale max=2

        if (not grid_ok) and (not out_ok):
            print(f"[WARN] No captures (grid/outlooker) in stage={s}, block={block_idx}")
            continue

        results.append({
            "stage": s,
            "block": block_idx,
            "seed": seed,
            "n_images": int(n),

            # sampling config (for reproducibility in logs)
            "grid_n_q": grid_n_q,
            "grid_exclude_border": grid_exclude_border,
            "grid_avg_over_groups": bool(grid_avg_over_groups),
            "out_n_xy": out_n_xy,
            "out_exclude_border": out_exclude_border,

            # GRID
            "MAD_grid_mean": grid_mean,        # norm if normalize_grid True
            "MAD_grid_std":  grid_std,
            "grid_Hf": Hf if grid_ok else None,
            "grid_Wf": Wf if grid_ok else None,
            "grid_denom": grid_denom,
            "MAD_grid_abs_mean": grid_abs_mean,  # abs featuremap L1

            # OUTLOOKER
            "MAD_outlook_mean": out_mean,        # norm in [0,1]
            "MAD_outlook_std":  out_std,
            "MAD_outlook_abs_mean": out_abs_mean 
       })

    cap_grid.close()
    cap_out.close()
    enable_mhsa_capture(model, False)
    return results


def _vals(rs, key):
    return [r[key] for r in rs if r.get(key, None) is not None]

def _mean_std(vals):
    if len(vals) == 0:
        return None, None, 0
    return float(np.mean(vals)), float(np.std(vals)), int(len(vals))


def print_mad_abs_by_stage_simple(all_res):
    by_stage = {}
    for r in all_res:
        by_stage.setdefault(r["stage"], []).append(r)

    print("\n=== MAD (ABS) by stage — simple view ===")
    print("GRID_abs is in featuremap L1 pixels; max = (Hf-1)+(Wf-1).")
    print("OUT_abs  is in 3×3 L1 steps; max = 2.\n")

    for s in sorted(by_stage.keys()):
        rs = by_stage[s]

        g_mu, g_sd, g_n = _mean_std(_vals(rs, "MAD_grid_abs_mean"))
        o_mu, o_sd, o_n = _mean_std(_vals(rs, "MAD_outlook_abs_mean"))

        denoms = sorted(set(_vals(rs, "grid_denom")))
        Hfs    = sorted(set(_vals(rs, "grid_Hf")))
        Wfs    = sorted(set(_vals(rs, "grid_Wf")))

        if len(denoms) == 1 and len(Hfs) == 1 and len(Wfs) == 1:
            scale = f"GRID max={denoms[0]:.0f} (Hf={Hfs[0]}, Wf={Wfs[0]}) | OUT max=2"
        else:
            scale = f"GRID max≈{(denoms[0] if len(denoms) else None)} | OUT max=2"

        def f2(mu, sd, n):
            if mu is None:
                return "None"
            return f"{mu:.2f}±{sd:.2f} (n={n})"

        print(f"stage {s}:  GRID_abs={f2(g_mu,g_sd,g_n)}   |   OUT_abs={f2(o_mu,o_sd,o_n)}   |   {scale}")
