from torchmetrics.functional import structural_similarity_index_measure as ssim_fn
import torch
import torch.nn.functional as F


def edge_loss(edge_pred, edge_label):
    return F.binary_cross_entropy(edge_pred, (edge_label > 0.5).float())


def ssim_loss(pred: torch.Tensor, gt:   torch.Tensor, data_range: float = 12.0) -> torch.Tensor:
    # gt has NaNs; build valid mask
    # pred, gt come in as float16 under autocast;

    p32 = pred.float()  # [B,1,H,W], float32
    g32 = gt.float()  # float32
    # mask out the holes by filling them with p32 (so SSIM=1 there)
    valid = ~torch.isnan(gt)
    g32 = torch.where(valid, g32, p32)

    # now compute SSIM in float32 outside of autocast
    with torch.cuda.amp.autocast(enabled=False):
        # run SSIM on CPU for deterministic matmul
        s = ssim_fn(p32, g32, data_range=data_range)

    # back to a loss in float16 to match other terms
    return (1.0 - s).to(pred.dtype)
def gradient_loss(pred, gt, sigma=0.1):
    # pred, gt: [B,H,W], in meters
    mask = ~torch.isnan(gt)
    dy_p = torch.abs(pred[:,1:] - pred[:,:-1])
    dx_p = torch.abs(pred[:,:,1:] - pred[:,:,:-1])
    dy_g = torch.abs(gt  [:,1:] - gt  [:,:-1])
    dx_g = torch.abs(gt  [:,:,1:] - gt  [:,:,:-1])

    m_y = mask[:,1:] & mask[:,:-1]
    m_x = mask[:,:,1:] & mask[:,:,:-1]

    ly = torch.abs(dy_p - dy_g)[m_y].mean()
    lx = torch.abs(dx_p - dx_g)[m_x].mean()
    return sigma * (ly + lx)

def masked_l1_loss(pred, gt):
    """
    pred, gt: tensors of the same shape, gt may contain nan where invalid.
    Returns the average L1 over only the non-nan entries of gt.
    """
    # absolute difference
    diff = torch.abs(pred - gt)
    # mask of valid pixels
    mask = ~torch.isnan(gt)
    # pick only valid entries and take their mean
    return diff[mask].mean()

def loss_funct(pred: torch.Tensor, label: torch.Tensor, epoch: int):
    label = label.unsqueeze(1)
    if pred.shape != label.shape:
        label = F.interpolate(label, size = (pred.shape[2], pred.shape[3]), mode='bilinear', align_corners=False)

    mask = ~torch.isnan(label)
    log_pred = torch.log(torch.clamp(pred, min=0.0) + 1e-3)
    log_gt_raw = torch.log(torch.clamp(label, min=0.0) + 1e-3)


    log_gt = torch.where(mask, log_gt_raw, log_pred)

    grad = gradient_loss(log_pred.squeeze(1), log_gt.squeeze(1), sigma=1)
    l1 = masked_l1_loss(log_pred[mask], log_gt[mask])
    batch_rng = (torch.nan_to_num(log_gt, nan=0.0).amax(dim=[1, 2, 3]) - torch.nan_to_num(log_gt, nan=0.0).amin(dim=[1, 2, 3])).mean().item()
    ssim = ssim_loss(log_pred, log_gt, data_range=batch_rng)
    edge_w = min(1.0, epoch / 40.0)  # grows from 0->1 during the first 40 epochs

    return l1 + edge_w *grad + 0.1*ssim