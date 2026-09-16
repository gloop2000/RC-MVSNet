# Depth-Anything-V2 edge prior for RC-MVSNet

An addition to RC-MVSNet that replaces the image-gradient weighting in the
unsupervised smoothness loss with a geometric edge prior derived from Depth
Anything V2, following the depth-edge idea in DVP-MVS (Yuan et al., 2024).

**Everything here is opt-in.** With no new flags the code takes its original
paths and the loss is bit-identical to the published implementation — verified,
not assumed. That matters because the baseline number in your results table has
to come from *this* tree, not from the paper.

---

## 1. The problem being fixed

`losses/modules.py::depth_smoothness` weights the depth-gradient penalty by the
RGB image gradient:

```python
weights_x = torch.exp(-(lambda_wt * torch.mean(torch.abs(image_dx), 3, keepdim=True)))
```

"Release the smoothness constraint where the image has an edge." This conflates
reflectance with geometry, and fails in both directions:

| situation | image gradient | what should happen | what the baseline does |
|---|---|---|---|
| painted stripe on a flat wall | strong | enforce smoothness | **releases** it — the network is free to invent a depth step |
| depth discontinuity between two similarly-coloured surfaces | ~none | release smoothness | **enforces** it — the depth blurs across the boundary |

A monocular depth prior separates the two, because it responds to geometry and
is blind to albedo. That is exactly the role DAv2 plays inside DVP-MVS.

Measured on synthetic cases with the actual code (both steps co-located at the
same column, so the weighting really applies):

```
                                          baseline    prior
albedo edge on flat wall (step spurious)    0.0023   0.1270   prior penalises 55x harder
textureless depth edge (step real)          0.6349   0.0116   prior releases  55x more
textured depth edge (both agree)            0.0116   0.0116   unchanged
```

## 2. What changed

| file | change |
|---|---|
| `losses/modules.py` | **new** `depth_smoothness_prior()`. `depth_smoothness()` untouched. |
| `losses/unsup_loss.py` | `UnSupLoss` / `UnsupLossMultiStage` take `edge_mode`, `lambda_edge`, `lambda_img`; `forward` takes an optional `edge`. The other `UnSupLoss_*` variants are untouched. |
| `datasets/dtu_train.py` | `MVSDataset(..., edge_prior_dir=None)`; `read_edge_prior()`; adds `sample["edge_prior"]` for the reference view only. |
| `train_rcmvsnet.py` | four new flags, passed to the dataset and the loss. |
| `tools/precompute_dav2_priors.py` | **new**, standalone. |

Three weighting modes, selected with `--edge_mode`:

- **`image`** (default) — the published loss. Baseline.
- **`prior`** — release factor `exp(-λ_e · E)` from the prior alone. Fixes both
  rows of the table above. This is the hypothesis under test.
- **`product`** — `exp(-λ_i · |∇I| · E)`: an image edge only releases smoothness
  where the prior agrees there is geometry. Fixes row 1 only. A conservative
  fallback if DAv2 turns out to miss real boundaries on DTU.

## 3. Precomputing the priors

DAv2 needs a DINOv2 backbone and a modern `transformers`; this repo pins
`torch==1.10.1` / Python 3.7. They will not co-install, and a ViT-Large inside
the dataloader would dominate epoch time anyway. So the priors are computed
once, offline, into PNGs.

```bash
conda create -n dav2 python=3.10 -y && conda activate dav2
pip install torch torchvision transformers pillow opencv-python numpy

# smoke test first: two scans, with preview strips to check the threshold
python tools/precompute_dav2_priors.py \
    --datapath /path/to/dtu --listfile lists/dtu/train.txt \
    --outdir /path/to/dtu_edge_priors \
    --limit 2 --save_preview

# then the full run
python tools/precompute_dav2_priors.py \
    --datapath /path/to/dtu --listfile lists/dtu/train.txt \
    --outdir /path/to/dtu_edge_priors
```

**Look at `_preview/` before the full run.** Each strip is `rgb | DAv2 depth |
edge`. You want thin closed contours on object boundaries and a clean interior.
If the interior is speckled, raise `--thresh`; if boundaries are broken, lower
it or raise `--close`.

One map per `(scan, view)`, not per lighting: the 7 DTU lightings of a view
share a camera and a scene, and the prior describes geometry. That makes the
precompute 7× cheaper (~3.9k maps, ~300 MB, rather than ~27k) and uses light
index 3, the well-exposed condition, so DAv2 sees the scene at its least
degraded. Worth one sentence in the report — a reader will wonder.

### The threshold is absolute, deliberately

`--thresh` is a threshold on the Roberts response of the **range-normalised**
depth: "the depth changes by more than `thresh` of the scene's depth range
across one pixel". A smooth surface spanning the frame gives ~1/640 ≈ 0.0016 per
pixel; an object/background step gives 0.1–0.5 even after DAv2's own smoothing.
The default 0.04 sits an order of magnitude clear of both.

An earlier draft normalised the gradient by its own 98th percentile. That is
wrong and testing caught it: it makes the threshold relative to whatever
gradient the scene contains, so a smoothly shaded surface with no discontinuity
at all lights up end to end — its gentle ramp *is* the 98th percentile.
Scale-invariance is what you want for depth; scale-invariance for the *gradient*
destroys the distinction the prior exists to draw.

## 4. Running the ablation

Fine-tune from the released checkpoint rather than training from scratch —
`pretrain/model_000014_cas.ckpt` is 14 epochs of compute you do not have to
repeat, and it makes the comparison a controlled one.

```bash
# A. baseline (no new flags — proves this tree reproduces the paper)
python train_rcmvsnet.py --trainpath $DTU --logdir ./log_baseline --resume ...

# B. the hypothesis
python train_rcmvsnet.py --trainpath $DTU --logdir ./log_prior --resume \
    --edge_prior_dir /path/to/dtu_edge_priors --edge_mode prior

# C. conservative variant
python train_rcmvsnet.py --trainpath $DTU --logdir ./log_product --resume \
    --edge_prior_dir /path/to/dtu_edge_priors --edge_mode product
```

Then `eval_rcmvsnet_dtu.py` → `matlab_eval/` for accuracy / completeness /
overall on DTU, and `eval_rcmvsnet_tanks.py` for T&T F1.

Run all three for the same number of epochs from the same checkpoint, same
seed. If you can only afford two, run A and B.

### What to report

The headline metric is the DTU overall score, but the *argument* lives in two
secondary numbers that speak directly to the mechanism:

- **`smooth_loss_stage3`** in TensorBoard, baseline vs. prior. It should be
  higher early in the prior run — the prior is refusing to let the network buy
  cheap loss reductions at albedo edges.
- **`thres2mm_error`** should improve more than `thres8mm_error`. The prior is a
  boundary-sharpness intervention, so it ought to move the tight threshold most.
  If the gains are uniform across 2/4/8 mm, something other than boundary
  behaviour is driving them and you should say so.

Also include qualitative crops at object boundaries. This is a sharpness claim;
a reader will want to see it.

### `--lambda_edge`

Controls how hard the prior releases smoothness: `exp(-λ_e)` at a confident
edge. Default 4.0 → weight 0.018, i.e. near-total release. If boundaries look
noisy, drop to 2.0 (weight 0.14). One short sweep over {2, 4, 8} on a couple of
scans is worth more than guessing.

## 5. Honest limitations

State these rather than let a reviewer find them:

- **This is an adaptation, not a reimplementation of DVP-MVS.** DVP-MVS uses its
  depth-edge prior to steer PatchMatch patch deformation inside an
  optimisation-based pipeline, and pairs it with a visibility prior that is not
  reproduced here. Borrowing the prior does not import the +8.34 T&T gap.
- **The prior enters only the smoothness term.** The cost volume, the depth
  hypothesis ranges, and the rendering-consistency branch are unchanged. The
  larger integration — using DAv2 to set per-pixel depth ranges in stage 1 of
  the cascade — is sketched in §6 and is future work.
- **DAv2 has no scale.** That is fine *here*, because the smoothness weight only
  cares where discontinuities are, not how deep they are. It is the reason this
  particular integration needs no scale/shift alignment, and worth saying
  explicitly — it is what makes the cheap version defensible.
- **Priors are computed on light 3 and shared.** Sound for geometry; an
  assumption nonetheless.

## 6. Not implemented: the depth-range hook

For the future-work section, with the specifics so it does not read as
hand-waving.

`CascadeMVSNet.forward` (`models/casmvsnet.py:191-209`) sets `cur_depth =
depth_values` at stage 1 — a `(B, D)` tensor. `get_depth_range_samples`
(`models/modules.py:569`) then takes its `dim()==2` branch: one global depth
range tiled identically to every pixel. Stages 2 and 3 pass a `(B, H, W)` map
and fall through to `get_cur_depth_range_samples`, which centres a per-pixel
window on the previous prediction.

Passing a `(B, H, W)` map at stage 1 needs no change to the sampling machinery
at all — the branch already exists. What it needs is metric alignment, and
RC-MVSNet supplies the bootstrap: `pseudo_depth` from sub-step S1 is already in
millimetres, so per-image `(a, b)` can be fitted by least squares between DAv2
inverse depth and `1/pseudo_depth` over the confident region.

Be precise about what that would buy: not scale, which still comes from the cost
volume, but **shape** — a smooth, complete, boundary-sharp depth field exactly
in the textureless regions where the cost volume is ambiguous and the uniform
range wastes most of its 48 hypotheses.
