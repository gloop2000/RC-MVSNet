# Running the experiments (Steps 3–6)

Commands for the development machine (JupyterHub, Linux). Everything is run from
the repository root `~/RC-MVSNet`. Read `EDGE_PRIOR.md` for the *why*; this file
is the *how*.

## 0. One-time setup

### 0.1 Paths used below

Every command below uses these variables (`$DTU`, `$RUNS`, ...). A variable
set with `export` only lives in the terminal where you typed it, so a new
terminal starts without them. The easy way is to save them once in a file:

```bash
cat > ~/RC-MVSNet/paths.sh <<'EOF'
export REPO=~/RC-MVSNet
export DTU=$REPO/dtu                  # Cameras, Depths, Depths_raw, Rectified
export DTU_TEST=$REPO/dtu_test        # scan1/{cams,images,pair.txt}, ...   <- adjust
export DTU_GT=$REPO/dtu_gt            # Points/stl/*.ply, ObsMask/*.mat     <- adjust
export PRIORS=$REPO/dtu_edge_priors
export RUNS=$REPO/runs
cd $REPO
EOF
```

Then, at the start of every new terminal, load it with:

```bash
source ~/RC-MVSNet/paths.sh
echo $DTU                              # should print /home/.../RC-MVSNet/dtu
```

`source` runs the file inside your current terminal, so the variables stay set
there. Running it as `bash paths.sh` or `./paths.sh` would not work: that starts
a separate shell, sets the variables there, and they vanish when it exits.

Keep the data out of git:

```bash
printf 'dtu/\ndtu_test/\ndtu_gt/\ndtu_edge_priors/\nruns/\n' >> .git/info/exclude
```

### 0.2 Get the synopsis code onto this machine

The edge-prior changes are committed as `develop: Implement synopsis` on
`github.com/gloop2000/RC-MVSNet`.

```bash
git pull origin main
git log --oneline -1                  # expect: develop: Implement synopsis
ls tools/precompute_dav2_priors.py EDGE_PRIOR.md
```

Copy `tools/eval_dtu_points.py` (delivered with this file) into `tools/` as well.

### 0.3 Environments (not the Python 2.7 env)

```bash
# training / evaluation
conda create -n rcmvsnet python=3.7 -y
source activate rcmvsnet              # or: conda activate rcmvsnet
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Depth Anything V2 priors only
conda create -n dav2 python=3.10 -y
source activate dav2
pip install torch torchvision "transformers>=4.45" pillow opencv-python numpy tqdm
```

### 0.4 Data checks

```bash
ls $DTU                               # Cameras  Depths  Depths_raw  Rectified
ls $DTU/Depths_raw | wc -l            # one folder per scan, no *_train
ls $DTU/Rectified/scan2_train | head -3
ls $DTU/Depths_raw/scan2 | head -3     # depth_map_0000.pfm, depth_visual_0000.png
ls $DTU_TEST/scan1                    # cams  images  pair.txt
ls $DTU_GT/Points/stl | head -3       # stl001_total.ply ...
ls $DTU_GT/ObsMask | head -3          # ObsMask1_10.mat  Plane1.mat ...
```

`DTU_TEST` is the separate *DTU testing* download from the README. `DTU_GT` is the
official DTU `SampleSet/MVS Data` folder with `Points.zip` extracted into it
(it must contain the `Points/stl` clouds for all 22 test scans, not only scan1/6).

**`ObsMask` is required for Step 6.** It holds, per scan, the observation mask
(`ObsMaskN_10.mat`: which part of space the structured-light scanner actually
saw) and the ground plane (`PlaneN.mat`). The official accuracy drops fused
points outside the mask, and completeness ignores GT points below the table
plane. Without them, background and table points count as errors and your
numbers are not comparable to any published DTU result.

Get it from the DTU MVS page (roboimagedata.compute.dtu.dk → MVS Data set 2014):

| download | provides |
|---|---|
| `SampleSet.zip` | `MVS Data/ObsMask/` for all scans, plus Points/Cleaned/Rectified for scan1 and scan6 only |
| `Points.zip` | `Points/stl/stlNNN_total.ply` for all scans |

Build `DTU_GT` from them:

```bash
mkdir -p $DTU_GT
unzip SampleSet.zip 'SampleSet/MVS Data/ObsMask/*' -d /tmp/dtu_sample
mv "/tmp/dtu_sample/SampleSet/MVS Data/ObsMask" $DTU_GT/
unzip Points.zip -d $DTU_GT            # must end up as $DTU_GT/Points/stl/...
ls $DTU_GT/ObsMask | grep -c ObsMask   # expect one per scan (>= 22)
```

If you already extracted `SampleSet` for the preprocessing work, just copy its
`MVS Data/ObsMask` folder instead of unzipping again.

### 0.5 Two gotchas in the training script

- **Single GPU: do not pass `--gpu`.** `--gpu` is parsed as a string, so
  `--gpu [0]` gives `len("[0]") == 3` and the script spawns 3 processes. Leave
  the default and select the card with `--true_gpu`.
- **Running two trainings at once** (different GPUs): give each its own
  `--master_port`, e.g. `11026` and `11027`.

---

## Step 3 — Reproduce the baseline (run A)

### 3.1 Evaluate the released checkpoint (sanity check, no training)

```bash
source activate rcmvsnet
python eval_rcmvsnet_dtu.py \
    --testpath $DTU_TEST --testlist lists/dtu/test.txt \
    --loadckpt pretrain/model_000014_cas.ckpt \
    --outdir $RUNS/pretrained_eval --true_gpu 0
```

Produces `$RUNS/pretrained_eval/mvsnetXXX_l3.ply` for the 22 scans.
`eval_rcmvsnet_dtu.py` defaults to `--true_gpu 1`; always pass `--true_gpu 0`
on a single-GPU machine. On out-of-memory add `--max_h 864 --max_w 1152`
(and use the same for every run). Score it with Step 6. The result should be
close to the RC-MVSNet paper's DTU table; if it is far off, fix the data before training.

### 3.2 Fine-tune the baseline

`--resume` loads the newest `model_*_cas.ckpt` / `model_*_nerf.ckpt` **from
`--logdir`**, so copy the released checkpoints there first. They are epoch 14,
so training starts at epoch 15 and `--epochs` must be **greater than 15**
(`--epochs 16` = one extra epoch, `--epochs 18` = three).

```bash
mkdir -p $RUNS/A_baseline
cp pretrain/model_000014_cas.ckpt pretrain/model_000014_nerf.ckpt $RUNS/A_baseline/

nohup python train_rcmvsnet.py \
    --dataset dtu_train --trainpath $DTU --testpath $DTU \
    --trainlist lists/dtu/train.txt --testlist lists/dtu/test.txt \
    --logdir $RUNS/A_baseline --resume --epochs 16 \
    --random_seed 1 --true_gpu 0 --master_port 11026 \
    > $RUNS/A_baseline/train.log 2>&1 &

tail -f $RUNS/A_baseline/train.log     # expect "resuming ... model_000014_cas.ckpt" and "start at epoch 15"
```

Check the seconds per iteration in the first log lines and multiply by the
iterations per epoch printed (`Iter-S1 x/N`) before deciding on `--epochs`.
A, B and C must use the **same** `--epochs` and `--random_seed`.

No edge-prior flags = the published loss, so this run is the baseline.

---

## Step 4 — Precompute the Depth Anything V2 edge priors

```bash
source activate dav2

# 4.1 smoke test: 2 scans + preview strips (rgb | DAv2 depth | edge)
python tools/precompute_dav2_priors.py \
    --datapath $DTU --listfile lists/dtu/train.txt \
    --outdir $PRIORS --limit 2 --save_preview

ls $PRIORS/_preview                    # open the PNGs in Jupyter's file browser
```

Look at the previews: thin closed contours on object boundaries, clean interiors.

- speckled interiors → raise `--thresh` (e.g. `0.06`)
- broken boundaries → lower `--thresh` (e.g. `0.03`) or raise `--close` (e.g. `5`)

After changing a parameter, rerun the smoke test with `--overwrite`.

```bash
# 4.2 full run (~3.9k maps, one per scan/view, from light 3)
python tools/precompute_dav2_priors.py \
    --datapath $DTU --listfile lists/dtu/train.txt \
    --outdir $PRIORS                   # add the --thresh/--close you settled on

ls $PRIORS | grep -c scan              # expect 79
ls $PRIORS/scan2 | wc -l               # expect 49
```

The first run downloads the ViT-L weights from Hugging Face. If the hub has no
internet, download `depth-anything/Depth-Anything-V2-Large-hf` elsewhere and
set `HF_HOME` to it. `--encoder vitb` is a faster fallback (say so in the report).

Only the training scans need priors: the prior is used in the training loss,
not at test time.

---

## Step 5 — Train the enhanced models (runs B and C)

Same starting checkpoint, epochs and seed as run A.

```bash
source activate rcmvsnet

# B: DAv2 geometric edges replace the image gradient (the hypothesis)
mkdir -p $RUNS/B_prior
cp pretrain/model_000014_cas.ckpt pretrain/model_000014_nerf.ckpt $RUNS/B_prior/
nohup python train_rcmvsnet.py \
    --dataset dtu_train --trainpath $DTU --testpath $DTU \
    --logdir $RUNS/B_prior --resume --epochs 16 \
    --random_seed 1 --true_gpu 0 --master_port 11027 \
    --edge_prior_dir $PRIORS --edge_mode prior --lambda_edge 4.0 \
    > $RUNS/B_prior/train.log 2>&1 &

# C: image gradient gated by the prior (conservative variant)
mkdir -p $RUNS/C_product
cp pretrain/model_000014_cas.ckpt pretrain/model_000014_nerf.ckpt $RUNS/C_product/
nohup python train_rcmvsnet.py \
    --dataset dtu_train --trainpath $DTU --testpath $DTU \
    --logdir $RUNS/C_product --resume --epochs 16 \
    --random_seed 1 --true_gpu 0 --master_port 11028 \
    --edge_prior_dir $PRIORS --edge_mode product \
    > $RUNS/C_product/train.log 2>&1 &
```

With one GPU, run them one after another (start the next when `train.log`
stops growing and `nvidia-smi` shows the card free). If time allows only two
runs, do A and B.

Optional `--lambda_edge` sweep (2, 4, 8): same command as B with
`--logdir $RUNS/B_prior_le2 --lambda_edge 2.0` etc. Use a short list (e.g. a
copy of `train.txt` with 10 scans, passed via `--trainlist`) and compare the
thres2mm numbers.

### Monitoring

```bash
tensorboard --logdir $RUNS --port 6006     # or the Jupyter TensorBoard extension
grep "thres2mm_error" $RUNS/B_prior/train.log | tail -3
```

Compare A vs B on `smooth_loss_stage3`, `thres2mm_error`, `thres8mm_error`
(see `EDGE_PRIOR.md` §4, "What to report").

---

## Step 6 — Evaluate

For each run, use the checkpoint from the last epoch
(`--epochs 16` → `model_000015_cas.ckpt`).

### 6.1 Depth maps + fused point clouds

```bash
source activate rcmvsnet
for RUN in A_baseline B_prior C_product; do
  CKPT=$(ls $RUNS/$RUN/model_*_cas.ckpt | sort | tail -1)
  echo "== $RUN  $CKPT"
  python eval_rcmvsnet_dtu.py \
      --testpath $DTU_TEST --testlist lists/dtu/test.txt \
      --loadckpt $CKPT --outdir $RUNS/$RUN/dtu_eval --true_gpu 0 \
      > $RUNS/$RUN/eval.log 2>&1
done
ls $RUNS/B_prior/dtu_eval/*.ply | wc -l      # expect 22
```

Keep all fusion flags at their defaults (`--prob_thres 0.8`,
`--num_consistency 3`) so the runs stay comparable. To redo only the fusion
step, add `--no_test`.

### 6.2 Accuracy, completeness, overall, F1

`tools/eval_dtu_points.py` mirrors the MATLAB protocol in `matlab_eval/`
(0.2 mm thinning, ObsMask, ground plane, 20 mm outlier cut) and adds
precision / recall / F1 at τ = 1, 2, 5 mm.

```bash
for RUN in pretrained_eval A_baseline/dtu_eval B_prior/dtu_eval C_product/dtu_eval; do
  echo "== $RUN"
  python tools/eval_dtu_points.py \
      --ply_dir $RUNS/$RUN --gt_dir $DTU_GT \
      --testlist lists/dtu/test.txt --thresholds 1,2,5
done
```

Each run writes `dtu_metrics.csv` (per scan + mean) next to its point clouds.
Expect a few minutes per scan (the 0.2 mm thinning is the slow part).

If MATLAB is available you can cross-check with the official scripts: edit
`gt_dataPath`, `dataPaths`, `resultsPaths` in `matlab_eval/BaseEvalMain_web_pt.m`
and `ComputeStat_web_pt.m`, and `resultPath` in `compute_mean.m`, then run
them in that order. Report which of the two evaluators your table uses.

### 6.3 Results table to fill in

| Run | edge_mode | Acc (mm) ↓ | Comp (mm) ↓ | Overall (mm) ↓ | F1@1 ↑ | F1@2 ↑ | F1@5 ↑ |
|---|---|---|---|---|---|---|---|
| Released ckpt (no fine-tune) | image | | | | | | |
| A baseline (fine-tuned) | image | | | | | | |
| B DAv2 prior | prior | | | | | | |
| C product | product | | | | | | |

Add qualitative crops at object boundaries (A vs B, same scan and view) from
`$RUNS/<run>/dtu_eval/scanX/depth_map/`.
