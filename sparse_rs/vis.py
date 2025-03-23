import numpy as np
import matplotlib
matplotlib.use('Agg')  # Important to disable interactive backend on a headless/cloud server
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import torch
import os
import argparse

from data.Im2GPS3k.download import load_im2gps_data


def unnormalize(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    Unnormalizes an image tensor: (C, H, W) and returns the result in [0, 1]
    """
    img = img.clone()
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(img, 0, 1)


def show_triplet(originals, adversarials, idx=0):
    """
    originals:   tensor of shape [N, 3, H, W], in [0,1]
    adversarials: same shape
    idx: which image to visualize
    Returns: a matplotlib Figure for saving
    """
    orig = originals[idx].detach().cpu()
    adv  = adversarials[idx].detach().cpu()

    if torch.equal(orig, adv): 
        return None

    diff_map = (adv - orig).abs().sum(dim=0)

    fig, axarr = plt.subplots(nrows=1, ncols=3, figsize=(12,4))

    # Original
    axarr[0].imshow(orig.permute(1,2,0).numpy())
    axarr[0].set_title("Original")
    axarr[0].axis('off')

    # Adversarial
    axarr[1].imshow(adv.permute(1,2,0).numpy())
    axarr[1].set_title("Adversarial")
    axarr[1].axis('off')

    # Difference (grayscale)
    axarr[2].imshow(diff_map.numpy(), cmap='gray')
    axarr[2].set_title("Difference Map")
    axarr[2].axis('off')

    fig.tight_layout()
    return fig


parser = argparse.ArgumentParser()
parser.add_argument('--adv_pth', type=str, default='',
                    help='Path to the .pth file containing adv results. Ex: results/sparse_rs_patches_*.pth')
parser.add_argument('--out_pdf', type=str, default='',
                    help='Filename for the output PDF.')
parser.add_argument('--num_samples', type=int, default=150,
                    help='How many samples to visualize.')
parser.add_argument('--unnormalize', action='store_true')
args = parser.parse_args()

if not args.adv_pth:
    args.adv_pth = (
        './results/sparse_rs_patches/'
        'sparse_rs_patches_geoclip_1_200_nqueries_1000_'
        'pinit_0.30_loss_margin_eps_20_targeted_False_'
        'targetclass_None_seed_42.pth'
    )

if not args.out_pdf:
    args.out_pdf = (
        './results/plots/sparse_rs_patches_geoclip_eps20.pdf'
    )



print("Loading adversarial results from:", args.adv_pth)
data = torch.load(args.adv_pth, map_location='cpu')
adv_images = data['adv'].cpu()
if 'qr' in data:
    qr = data['qr'].cpu()
else:
    qr = torch.arange(1, adv_images.shape[0] + 1)

print("Adversarial image tensor shape:", adv_images.shape)


data_path = "./data"
x_test, y_test = load_im2gps_data(data_path)
print("Original x_test shape:", x_test.shape)

assert x_test.shape == adv_images.shape, \
       f"Mismatch in shapes: x_test={x_test.shape}, adv={adv_images.shape}."



x_test_unnorm_list = [unnormalize(x_test[i]) for i in range(x_test.size(0))]
x_test_unnorm = torch.stack(x_test_unnorm_list)

adv_unnorm_list = [unnormalize(adv_images[i]) for i in range(adv_images.size(0))]
adv_unnorm = torch.stack(adv_unnorm_list)



nqueries = 10000
valid_mask = (qr >= 0) & (qr <= nqueries)
valid_inds = valid_mask.nonzero().squeeze()

qvals = qr[valid_inds]
qvals_sorted, sort_idx = qvals.sort(descending=True)
valid_inds_sorted = valid_inds[sort_idx]

sample_inds = valid_inds_sorted[:-1].tolist()

if not sample_inds:
    print("No valid samples to visualize! Check your query thresholds.")
    exit()



print(f"Saving up to {len(sample_inds)} images in {args.out_pdf} ...")
with PdfPages(args.out_pdf) as pdf:
    for i, idx in enumerate(sample_inds):
        orig_img_batch = x_test_unnorm[idx].unsqueeze(0)
        adv_img_batch = adv_unnorm[idx].unsqueeze(0)

        fig = show_triplet(orig_img_batch, adv_img_batch, idx=0)
        if fig == None:
            continue

        fig.suptitle(f"Sample {idx} — QR: {qr[idx].item()}", fontsize=14)

        pdf.savefig(fig)
        plt.close(fig)

print(f"Done. PDF saved to: {args.out_pdf}")
