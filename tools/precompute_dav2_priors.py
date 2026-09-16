#!/usr/bin/env python3
"""Precompute Depth-Anything-V2 geometric edge priors for the DTU training set.

Why offline
-----------
RC-MVSNet pins torch==1.10.1 / Python 3.7. Depth Anything V2 needs a DINOv2
backbone and a modern timm/transformers. The two will not co-install, and
running a ViT-Large inside the dataloader would dominate epoch time anyway. So
this script runs in its OWN environment, writes PNGs to disk, and the training
job just reads them.

    conda create -n dav2 python=3.10 -y && conda activate dav2
    pip install torch torchvision transformers pillow opencv-python numpy tqdm
    python tools/precompute_dav2_priors.py --datapath /path/to/dtu \
        --listfile lists/dtu/train.txt --outdir /path/to/dtu_edge_priors

What it produces
----------------
    <outdir>/<scan>/edge_<view:04d>.png      uint8, 512x640, 0..255 == 0..1

One map per (scan, view), NOT per lighting. The 7 DTU lightings of a view share
a camera and a scene; only the illumination differs, and the prior describes
geometry. Computing on one lighting is 7x cheaper in both time and disk
(~3.9k maps, ~300 MB, instead of ~27k maps) and is the more faithful input:
light index 3 is the well-exposed condition, so DAv2 sees the scene at its
least degraded. Pass --light to change which one is used.

Pipeline (adapted from DVP-MVS, Yuan et al. 2024, sec. 3.2)
-----------------------------------------------------------
    1. DAv2 -> relative inverse depth, resized to 512x640
    2. robust per-image normalisation to [0, 1]  (1st/99th percentile)
    3. Roberts cross operator -> gradient magnitude
    4. robust normalisation + threshold -> binary edge mask
    5. erosion-dilation alignment -> closes 1px gaps, thickens the boundary so
       the released band survives slight prior/image misalignment
    6. Gaussian blur -> soft weights in [0, 1]

Step 5 is what DVP-MVS calls the erosion-dilation strategy for "fine-grained
homogeneous boundaries". Use --no_morph to ablate it.

Sanity-check a handful of scans with --limit and --save_preview before
committing to the full run.
"""

import argparse
import os
import sys

import cv2
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Depth Anything V2
# ---------------------------------------------------------------------------

HF_CHECKPOINTS = {
    'vits': 'depth-anything/Depth-Anything-V2-Small-hf',
    'vitb': 'depth-anything/Depth-Anything-V2-Base-hf',
    'vitl': 'depth-anything/Depth-Anything-V2-Large-hf',
}


class DepthAnythingV2Runner(object):
    """Thin wrapper over the HuggingFace port of Depth Anything V2.

    Returns RELATIVE INVERSE depth (larger == nearer), which is all this script
    needs: every downstream step is scale- and shift-invariant because it only
    looks at gradients after a per-image normalisation.
    """

    def __init__(self, encoder='vitl', device='cuda'):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.torch = torch
        self.device = device
        ckpt = HF_CHECKPOINTS[encoder]
        print('[dav2] loading {}'.format(ckpt))
        self.processor = AutoImageProcessor.from_pretrained(ckpt)
        self.model = AutoModelForDepthEstimation.from_pretrained(ckpt).to(device).eval()

    def __call__(self, rgb_uint8):
        """rgb_uint8: HxWx3 uint8 RGB.  ->  float32 HxW relative inverse depth."""
        torch = self.torch
        h, w = rgb_uint8.shape[:2]
        inputs = self.processor(images=Image.fromarray(rgb_uint8), return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            pred = self.model(**inputs).predicted_depth        # [1, h', w']
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1), size=(h, w), mode='bicubic', align_corners=False)
        return pred.squeeze().detach().cpu().float().numpy()


# ---------------------------------------------------------------------------
# depth -> edge probability
# ---------------------------------------------------------------------------

def robust_normalise(x, lo_pct=1.0, hi_pct=99.0):
    """Min-max to [0, 1] using percentiles, so one outlier pixel cannot flatten
    the whole map."""
    lo = np.percentile(x, lo_pct)
    hi = np.percentile(x, hi_pct)
    if hi - lo < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def roberts_magnitude(x):
    """Roberts cross operator, the one DVP-MVS uses.

    Two 2x2 diagonal kernels. Cheap, and its tight support keeps the response
    localised on the boundary instead of smearing it over several pixels the
    way Sobel does -- which matters here because the response is about to be
    dilated deliberately.
    """
    kx = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.float32)
    ky = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    gx = cv2.filter2D(x, cv2.CV_32F, kx)
    gy = cv2.filter2D(x, cv2.CV_32F, ky)
    return np.sqrt(gx * gx + gy * gy)


def depth_to_edge(depth_rel, thresh=0.04, dilate=3, close=3, blur=3,
                  use_morph=True):
    """Relative (inverse) depth map -> geometric-edge probability in [0, 1].

    `thresh` is an ABSOLUTE threshold on the Roberts response of the
    range-normalised depth, i.e. "the depth changes by more than `thresh` of the
    scene's full depth range across one pixel".

    That absolute reading is deliberate. An earlier version normalised the
    gradient by its own 98th percentile, which makes the threshold relative to
    whatever gradient the scene happens to contain -- so a smoothly shaded
    surface with no discontinuity at all lights up end to end, because its
    gentle ramp IS the 98th percentile. Scale-free is what we want for depth;
    scale-free for the GRADIENT destroys the very distinction being drawn.

    For reference: a smooth surface spanning the frame gives ~1/640 = 0.0016
    per pixel, while an object/background step gives 0.1-0.5 even after DAv2's
    own smoothing. The default sits an order of magnitude clear of both.
    """
    d = robust_normalise(depth_rel)
    g = roberts_magnitude(d)

    if not use_morph:
        # Soft response, same threshold semantics: g == thresh maps to 1.0.
        return np.clip(g / max(thresh, 1e-8), 0.0, 1.0).astype(np.float32)

    binary = (g > thresh).astype(np.uint8)

    # Close first: joins boundary fragments broken by DAv2's own smoothing.
    if close > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)

    # Then dilate: widens the released band so a boundary that sits a pixel or
    # two off in the prior still covers the true discontinuity. Erring wide is
    # the safe direction -- a slightly too-wide release costs a little
    # smoothing, a too-narrow one lets the depth blur straight across the edge.
    if dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
        binary = cv2.dilate(binary, k)

    e = binary.astype(np.float32)
    if blur > 0:
        ksz = blur if blur % 2 == 1 else blur + 1
        e = cv2.GaussianBlur(e, (ksz, ksz), 0)

    return np.clip(e, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def read_scans(listfile):
    with open(listfile) as f:
        return [ln.strip() for ln in f if ln.strip()]


def num_views(datapath):
    with open(os.path.join(datapath, 'Cameras', 'pair.txt')) as f:
        return int(f.readline())


def save_preview(path, rgb, depth_rel, edge):
    d = (robust_normalise(depth_rel) * 255).astype(np.uint8)
    d = cv2.applyColorMap(d, cv2.COLORMAP_INFERNO)
    e = cv2.cvtColor((edge * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, np.concatenate([bgr, d, e], axis=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--datapath', required=True,
                    help='DTU root containing Cameras/ Depths/ Depths_raw/ Rectified/')
    ap.add_argument('--listfile', default='lists/dtu/train.txt')
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--encoder', default='vitl', choices=list(HF_CHECKPOINTS))
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--light', type=int, default=3,
                    help='DTU lighting index the prior is computed from (0-6)')

    ap.add_argument('--thresh', type=float, default=0.04,
                    help='edge threshold: fraction of the scene depth range a '
                         'step must cross in one pixel to count as geometry')
    ap.add_argument('--dilate', type=int, default=3)
    ap.add_argument('--close', type=int, default=3)
    ap.add_argument('--blur', type=int, default=3)
    ap.add_argument('--no_morph', action='store_true',
                    help='skip the erosion-dilation stage (ablation)')

    ap.add_argument('--limit', type=int, default=0,
                    help='stop after N scans; use for a smoke test')
    ap.add_argument('--save_preview', action='store_true',
                    help='also write rgb|depth|edge strips to <outdir>/_preview')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    scans = read_scans(args.listfile)
    if args.limit:
        scans = scans[:args.limit]
    nviews = num_views(args.datapath)
    print('[dav2] {} scans x {} views, lighting {}'.format(len(scans), nviews, args.light))

    runner = DepthAnythingV2Runner(args.encoder, args.device)

    if args.save_preview:
        os.makedirs(os.path.join(args.outdir, '_preview'), exist_ok=True)

    total = done = skipped = 0
    for scan in scans:
        outscan = os.path.join(args.outdir, scan)
        os.makedirs(outscan, exist_ok=True)

        for vid in range(nviews):
            total += 1
            outpath = os.path.join(outscan, 'edge_{:0>4}.png'.format(vid))
            if os.path.exists(outpath) and not args.overwrite:
                skipped += 1
                continue

            # NOTE the +1: image files are 1-indexed, everything else is 0-indexed.
            imgpath = os.path.join(
                args.datapath,
                'Rectified/{}_train/rect_{:0>3}_{}_r5000.png'.format(scan, vid + 1, args.light))
            if not os.path.exists(imgpath):
                print('[dav2] missing image, skipping: {}'.format(imgpath))
                continue

            rgb = cv2.cvtColor(cv2.imread(imgpath), cv2.COLOR_BGR2RGB)
            if rgb.shape[:2] != (512, 640):
                raise RuntimeError(
                    '{} is {}x{}, expected 512x640. This script assumes the '
                    'MVSNet-preprocessed Rectified images.'.format(
                        imgpath, rgb.shape[0], rgb.shape[1]))

            depth_rel = runner(rgb)
            edge = depth_to_edge(depth_rel,
                                 thresh=args.thresh, dilate=args.dilate,
                                 close=args.close, blur=args.blur,
                                 use_morph=not args.no_morph)

            Image.fromarray((edge * 255).astype(np.uint8)).save(outpath)
            done += 1

            if args.save_preview and vid == 0:
                save_preview(os.path.join(args.outdir, '_preview',
                                          '{}_{:0>4}.jpg'.format(scan, vid)),
                             rgb, depth_rel, edge)

            if done % 200 == 0:
                print('[dav2] {}/{} written'.format(done, total))

        print('[dav2] {} done'.format(scan))

    print('[dav2] finished: {} written, {} already present, {} total'.format(
        done, skipped, total))


if __name__ == '__main__':
    sys.exit(main())
