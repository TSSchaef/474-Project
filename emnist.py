import numpy as np
import pandas as pd
import os

def load_emnist_data(split="train"):
    assert split in ["train", "test"], "Invalid split. Choose 'train' or 'test'."
    
    file_path = os.path.join("emnist", f"emnist-balanced-{split}.csv")
    data = pd.read_csv(file_path, header=None).values
    
    labels = data[:, 0].astype(np.int64)
    images = data[:, 1:].astype(np.float32).reshape(-1, 28, 28)

    return images, labels

def extract_training_samples():
    return load_emnist_data("train")

def extract_test_samples():
    return load_emnist_data("test")

