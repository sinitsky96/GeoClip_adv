import os
import re
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')  # Important for headless servers
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import linear_sum_assignment

from data.mixed_dataset.download import get_mixed_dataloader, get_transforms as get_mixed_transforms

def parse_run_params(path_str):
    """
    Extract (model, eps_str, k_str, targeted_str) from path name.
    - 'model' => 'clip' or 'geoclip'
    - 'eps_str' => 'sparse' if eps=150, else numeric string
    - 'k_str'   => integer as string
    - 'targeted_str' => 'True' or 'False'
    """
    # Identify model
    if "_clip_" in path_str and "_geoclip_" not in path_str:
        model = "clip"
    elif "_geoclip_" in path_str:
        model = "geoclip"
    else:
        raise ValueError(f"Could not parse model from: {path_str}")

    # Identify eps
    eps_match = re.search(r"_eps_(\d+)_", path_str)
    if not eps_match:
        raise ValueError(f"Could not find eps in: {path_str}")
    eps_val = int(eps_match.group(1))
    eps_str = "sparse" if (eps_val == 150) else str(eps_val)

    # Identify k
    k_match = re.search(r"_k_(\d+)_", path_str)
    if not k_match:
        raise ValueError(f"Could not find k in: {path_str}")
    k_str = k_match.group(1)

    # Identify targeted
    if "_targeted_True" in path_str:
        targeted_str = "True"
    elif "_targeted_False" in path_str:
        targeted_str = "False"
    else:
        raise ValueError(f"Could not parse targeted info from: {path_str}")

    return (model, eps_str, k_str, targeted_str)


def build_cost_matrix(clean_imgs, adv_imgs, threshold=1e-5):
    """
    Pixel-based difference cost for Hungarian alignment.
    clean_imgs, adv_imgs => shape [N, 3, H, W]
    """
    N = clean_imgs.size(0)
    cost = torch.zeros((N, N), dtype=torch.float)
    for i in range(N):
        diff_i = (clean_imgs[i] - adv_imgs)  # shape [N, 3, H, W]
        abs_diff_i = diff_i.abs() > threshold
        cost[i] = abs_diff_i.any(dim=1).any(dim=1).float().sum(dim=1)
    return cost

def align_adversarial(clean_imgs, adv_imgs, threshold=1e-5):
    """
    Reorder adv_imgs so adv[i] matches clean_imgs[i].
    """
    cost = build_cost_matrix(clean_imgs, adv_imgs, threshold=threshold)
    row_ind, col_ind = linear_sum_assignment(cost.numpy())
    adv_aligned = adv_imgs.clone()
    for i, c in zip(row_ind, col_ind):
        adv_aligned[i] = adv_imgs[c]
    return adv_aligned

def unnormalize(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    Unnormalize a (3,H,W) tensor from typical stats => [0,1].
    """
    x = img.clone()
    for t, m, s in zip(x, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(x, 0, 1)

def save_image_as_pdf(tensor_img, pdf_path):
    """
    Saves a single image (no side-by-side) to a 1-page PDF.
    """
    fig = plt.figure()
    plt.imshow(tensor_img.permute(1,2,0).cpu().numpy())
    plt.axis('off')
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig)
    plt.close(fig)


if __name__ == "__main__":
    data_path = "./data"
    transform = get_mixed_transforms()
    dataloader = get_mixed_dataloader(
        data_path,
        batch_size=150,
        samples_per_dataset=75,
        transform=transform,
        clip_varient=False
    )

    x_tensors, y_tensors = [], []
    for x, y in dataloader:
        x_tensors.append(x)
        y_tensors.append(y)

    x_test = torch.cat(x_tensors, dim=0)  # [N,3,H,W]
    y_test = torch.cat(y_tensors, dim=0)  # [N,2] or similar
    print("Loaded dataset shapes:", x_test.shape, y_test.shape)

    N = x_test.size(0)

    root_output_dir = "per_image_dirs_single"
    os.makedirs(root_output_dir, exist_ok=True)

    # For each image i, create a sub-folder and store "clean.pdf"
    for i_idx in range(N):
        out_dir_i = os.path.join(root_output_dir, f"image_{i_idx}")
        os.makedirs(out_dir_i, exist_ok=True)

        # Save the clean image as "clean.pdf"
        clean_unnorm = unnormalize(x_test[i_idx])
        clean_pdf_path = os.path.join(out_dir_i, "clean.pdf")
        save_image_as_pdf(clean_unnorm, clean_pdf_path)


    adv_paths = [
        # L0 - clip
        "results/........",
    ]


    for path_pth in adv_paths:
        print(f"Loading adversarial from: {path_pth}")

        model_str, eps_str, k_str, tgt_str = parse_run_params(path_pth)

        data = torch.load(path_pth, map_location='cpu')
        adv = data['adv'].cpu()
        assert adv.shape == x_test.shape, (
            f"Mismatch in shapes: adv={adv.shape}, x_test={x_test.shape}"
        )

        adv_aligned = align_adversarial(x_test, adv, threshold=1e-5)

        for i_idx in range(N):
            out_dir_i = os.path.join(root_output_dir, f"image_{i_idx}")
            adv_i = unnormalize(adv_aligned[i_idx])
            
            pdf_filename = f"model_{model_str}_eps_{eps_str}_k_{k_str}_targeted_{tgt_str}.pdf"
            pdf_path = os.path.join(out_dir_i, pdf_filename)

            # Save single image in that PDF
            save_image_as_pdf(adv_i, pdf_path)

    print("Done! Each 'image_i' folder has:")
    print("  - a clean.pdf for the clean version")
    print("  - a single-page PDF for each .pth file’s adversarial version")
