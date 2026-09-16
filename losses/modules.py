import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)
        self.mask_pool = nn.AvgPool2d(3, 1)
        # self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y, mask):
        # print('mask: {}'.format(mask.shape))
        # print('x: {}'.format(x.shape))
        # print('y: {}'.format(y.shape))
        x = x.permute(0, 3, 1, 2)  # [B, H, W, C] --> [B, C, H, W]
        y = y.permute(0, 3, 1, 2)
        mask = mask.permute(0, 3, 1, 2)

        # x = self.refl(x)
        # y = self.refl(y)
        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)
        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y
        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)
        SSIM_mask = self.mask_pool(mask)
        output = SSIM_mask * torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)
        return output.permute(0, 2, 3, 1)  # [B, C, H, W] --> [B, H, W, C]


def gradient_x(img):
    return img[:, :, :-1, :] - img[:, :, 1:, :]

def gradient_y(img):
    return img[:, :-1, :, :] - img[:, 1:, :, :]

def gradient(pred):
    D_dy = pred[:, 1:, :, :] - pred[:, :-1, :, :]
    D_dx = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    return D_dx, D_dy


def depth_smoothness(depth, img,lambda_wt=1):
    """Computes image-aware depth smoothness loss."""
    # print('depth: {} img: {}'.format(depth.shape, img.shape))
    depth_dx = gradient_x(depth)
    depth_dy = gradient_y(depth)
    image_dx = gradient_x(img)
    image_dy = gradient_y(img)
    weights_x = torch.exp(-(lambda_wt * torch.mean(torch.abs(image_dx), 3, keepdim=True)))
    weights_y = torch.exp(-(lambda_wt * torch.mean(torch.abs(image_dy), 3, keepdim=True)))
    # print('depth_dx: {} weights_x: {}'.format(depth_dx.shape, weights_x.shape))
    # print('depth_dy: {} weights_y: {}'.format(depth_dy.shape, weights_y.shape))
    smoothness_x = depth_dx * weights_x
    smoothness_y = depth_dy * weights_y
    return torch.mean(torch.abs(smoothness_x)) + torch.mean(torch.abs(smoothness_y))


def compute_reconstr_loss(warped, ref, mask, simple=True):
    if simple:
        return F.smooth_l1_loss(warped*mask, ref*mask, reduction='mean')
    else:
        alpha = 0.5
        ref_dx, ref_dy = gradient(ref * mask)
        warped_dx, warped_dy = gradient(warped * mask)
        photo_loss = F.smooth_l1_loss(warped*mask, ref*mask, reduction='mean')
        grad_loss = F.smooth_l1_loss(warped_dx, ref_dx, reduction='mean') + \
                    F.smooth_l1_loss(warped_dy, ref_dy, reduction='mean')
        return (1 - alpha) * photo_loss + alpha * grad_loss

# ---------------------------------------------------------------------------
# Edge-prior-aware depth smoothness.
#
# `depth_smoothness` above weights the depth-gradient penalty by the RGB image
# gradient. That conflates two different things:
#
#   * albedo edges (a painted stripe on a flat wall) produce a strong image
#     gradient and wrongly RELEASE the smoothness constraint;
#   * geometric edges between two similarly-coloured surfaces produce almost no
#     image gradient and are wrongly OVER-SMOOTHED.
#
# DVP-MVS (Yuan et al., 2024) attacks the same problem by deriving edges from a
# monocular depth prior (Depth Anything V2) instead of from image intensity.
# `depth_smoothness_prior` adapts that idea to this unsupervised loss: the
# release factor comes from a precomputed geometric-edge probability map rather
# than from the image itself.
#
# NOTE: this is an adaptation, not a reimplementation. DVP-MVS uses its
# depth-edge prior to steer PatchMatch patch deformation inside an
# optimisation-based pipeline; here the same prior re-weights a smoothness term
# in a learned unsupervised loss.
# ---------------------------------------------------------------------------

def depth_smoothness_prior(depth, img, edge, lambda_img=1.0, lambda_edge=4.0,
                           mode='prior'):
    """Depth smoothness whose release factor can come from a geometric edge prior.

    Args:
        depth:       [B, H, W, 1] predicted depth for the reference view.
        img:         [B, H, W, C] reference image, exactly as the baseline uses.
        edge:        [B, H, W, 1] geometric-edge probability in [0, 1], or None.
        lambda_img:  weight on the image-gradient release factor.
        lambda_edge: weight on the edge-prior release factor.
        mode:
            'image'   -- identical to `depth_smoothness`; the baseline.
            'prior'   -- release smoothness only where the prior says there is a
                         genuine depth discontinuity. Fixes BOTH failure cases
                         above and is the hypothesis under test.
            'product' -- image gradient gated by the prior: an image edge only
                         releases smoothness where the prior agrees. Fixes the
                         albedo-edge case only; a conservative middle ground if
                         the prior turns out to miss real boundaries.

    Returns:
        Scalar tensor, same scale and sign convention as `depth_smoothness`.
    """
    depth_dx = gradient_x(depth)
    depth_dy = gradient_y(depth)

    if edge is None or mode == 'image':
        # Baseline path. Kept here so a single call site can serve the ablation.
        image_dx = gradient_x(img)
        image_dy = gradient_y(img)
        weights_x = torch.exp(-(lambda_img * torch.mean(torch.abs(image_dx), 3, keepdim=True)))
        weights_y = torch.exp(-(lambda_img * torch.mean(torch.abs(image_dy), 3, keepdim=True)))

    else:
        # Co-locate the edge probability with the finite difference: the value
        # for the difference between pixels i and i+1 is the mean of the two.
        edge_x = 0.5 * (edge[:, :, :-1, :] + edge[:, :, 1:, :])
        edge_y = 0.5 * (edge[:, :-1, :, :] + edge[:, 1:, :, :])

        if mode == 'prior':
            weights_x = torch.exp(-(lambda_edge * edge_x))
            weights_y = torch.exp(-(lambda_edge * edge_y))

        elif mode == 'product':
            image_dx = torch.mean(torch.abs(gradient_x(img)), 3, keepdim=True)
            image_dy = torch.mean(torch.abs(gradient_y(img)), 3, keepdim=True)
            weights_x = torch.exp(-(lambda_img * image_dx * edge_x))
            weights_y = torch.exp(-(lambda_img * image_dy * edge_y))

        else:
            raise ValueError(
                "depth_smoothness_prior: mode must be 'image', 'prior' or "
                "'product', got {!r}".format(mode))

    smoothness_x = depth_dx * weights_x
    smoothness_y = depth_dy * weights_y
    return torch.mean(torch.abs(smoothness_x)) + torch.mean(torch.abs(smoothness_y))
