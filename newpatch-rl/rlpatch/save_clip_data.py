import argparse
import os
import joblib
import torch

from data.Im2GPS3k.download import load_im2gps_data, CLIP_load_data_tensor


def save_new_data(data_dir, output_filename):
    """
    Loads the dataset (already transformed by download.py),
    and saves the results in the same style as create_new_ens.py.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # if we run embeddings

    print(f"Loading data from {data_dir} ...")
    X, y = load_im2gps_data(data_dir)
    print(f"Finished loading. X shape = {X.shape}, y shape = {y.shape}")
    save_dir = "stmodels"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    path_to_save = os.path.join(save_dir, output_filename)
    print(f"Saving tensors to {path_to_save} ...")
    joblib.dump(X, path_to_save)
    print("Done saving!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='.', help='Root path for your dataset')
    parser.add_argument('--output_file', type=str, default='geo_data.pkl', help='Filename for the saved output')
    args = parser.parse_args()

    save_new_data(args.data_dir, args.output_file, args.batch_size)


