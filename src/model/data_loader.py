# data_loader.py
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import yaml  # Added for YAML parsing
import cv2
import torchvision.transforms.functional as TF
import numpy as np

# Import configuration
from . import config


# PyTorch Dataset class
class DrivingDataset(Dataset):
    def __init__(self, samples_list, img_dir, transform=None):
        """
        Args:
            samples_list (list): List of tuples (image_filename, steering_angle_float).
            img_dir (string): Directory with images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_dir = img_dir
        self.transform = transform
        self.samples = samples_list

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, steering_angle = self.samples[idx]
        img_path = os.path.join(self.img_dir, img_name)

        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"WARNING: Image {img_path} not found, skipping sample.")
            return None, None  # Will be handled by collate_fn
        except Exception as e:
            print(f"WARNING: Error loading image {img_path}: {e}, skipping sample.")
            return None, None

        if self.transform:
            image = self.transform(image)

        if image is None:  # Additional safeguard
            return None, None

        return image, torch.tensor([steering_angle], dtype=torch.float32)


def crop_top_and_bottom(img):
    width, height = img.size
    top_crop = int(height * 0.4)
    bottom_crop = int(height * 0.1)
    return img.crop((0, top_crop, width, height - bottom_crop))  # (left, upper, right, lower)

def apply_sobel_edge(image_tensor):
    """
    Applies Sobel filter to emphasize edges. Operates on a PyTorch tensor (C x H x W).
    Returns a tensor with the same shape and dtype.
    """
    # Convert to grayscale: assume input is in range [-0.5, 0.5]
    grayscale = TF.rgb_to_grayscale(image_tensor + 0.5)  # shift back to [0, 1] for OpenCV compatibility

    # Convert to numpy and scale to 0-255 uint8
    grayscale_np = grayscale.squeeze(0).numpy() * 255.0
    grayscale_np = grayscale_np.astype(np.uint8)

    # Apply Sobel filter
    sobel_x = cv2.Sobel(grayscale_np, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(grayscale_np, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    sobel = np.clip(sobel / sobel.max(), 0, 1)

    # Convert back to tensor and expand to 1 channel, then repeat to match RGB shape
    sobel_tensor = torch.tensor(sobel, dtype=torch.float32).unsqueeze(0)  # 1 x H x W
    sobel_rgb = sobel_tensor.repeat(3, 1, 1)  # 3 x H x W to match input channels

    return sobel_rgb - 0.5  # back to [-0.5, 0.5]

# Definition of image transformations
def get_data_transforms():
    """Returns a dictionary of transformations for training and validation sets."""
    return {
        "train": transforms.Compose(
            [
                transforms.Lambda(crop_top_and_bottom),
                transforms.Resize((config.IMG_HEIGHT, config.IMG_WIDTH)),
                transforms.ToTensor(),  # Scales to [0.0, 1.0] and changes to C x H x W
                transforms.Lambda(lambda x: x - 0.5),  # Scales to [-0.5, 0.5]
                transforms.Lambda(apply_sobel_edge),
                # Augmentations for the training set can be added here, e.g.:
                # transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Lambda(crop_top_and_bottom),
                transforms.Resize((config.IMG_HEIGHT, config.IMG_WIDTH)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x - 0.5),  # Scales to [-0.5, 0.5]
                transforms.Lambda(apply_sobel_edge),
            ]
        ),
    }


# collate_fn function to skip broken samples
def collate_fn_skip_broken(batch):
    """Filters out samples that returned None from Dataset.__getitem__."""
    # Remove (None, None) from the batch
    batch = list(filter(lambda x: x[0] is not None and x[1] is not None, batch))
    if not batch:  # If the entire batch is empty after filtering
        return torch.tensor([]), torch.tensor([])
    return torch.utils.data.dataloader.default_collate(batch)


# Function to parse the annotations file
def parse_annotations_file(annotations_filepath):
    all_samples = []
    if not os.path.exists(annotations_filepath):
        print(f"ERROR: Annotations file '{annotations_filepath}' not found.")
        return all_samples

    with open(annotations_filepath, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue  # Skip empty lines

            parts = line.split(" ", 1)  # Split only on the first space
            if len(parts) < 2:
                print(
                    f"WARNING: Line {line_num} in '{annotations_filepath}' has incorrect format: '{line}', skipping."
                )
                continue

            img_name = parts[0]
            try:
                # Steering angle is the first value before the comma
                steering_angle_str = parts[1].split(",")[0]
                steering_angle = float(steering_angle_str)
                all_samples.append((img_name, steering_angle))
            except ValueError:
                print(
                    f"WARNING: Line {line_num} in '{annotations_filepath}', cannot convert angle to float: '{parts[1]}', skipping."
                )
            except IndexError:
                print(
                    f"WARNING: Line {line_num} in '{annotations_filepath}', incorrect format after comma: '{parts[1]}', skipping."
                )
    return all_samples


# Function to load YAML configuration
def load_yaml_config(yaml_filepath):
    """Loads configuration from a YAML file."""
    if not os.path.exists(yaml_filepath):
        print(f"ERROR: YAML configuration file '{yaml_filepath}' not found.")
        return None
    try:
        with open(yaml_filepath, "r") as f:
            config_data = yaml.safe_load(f)
        return config_data
    except yaml.YAMLError as e:
        print(f"ERROR: Could not parse YAML file '{yaml_filepath}': {e}")
        return None
    except Exception as e:
        print(
            f"ERROR: An unexpected error occurred while reading YAML file '{yaml_filepath}': {e}"
        )
        return None


# Refactored function to create DataLoaders
def get_dataloaders(
    samples_for_train_val_split,
    dedicated_test_samples,
    images_dir,
    data_transforms,
    batch_size=None,
    val_split_size=None,
    random_state=None,
    num_workers=None,
):
    """Creates and returns DataLoaders for training, validation, and test sets."""
    # Use provided params or fallback to config
    current_batch_size = batch_size if batch_size is not None else config.BATCH_SIZE
    current_val_split_size = (
        val_split_size if val_split_size is not None else config.VAL_SPLIT_SIZE
    )
    current_random_state = (
        random_state if random_state is not None else config.RANDOM_STATE
    )
    current_num_workers = num_workers if num_workers is not None else config.NUM_WORKERS

    # Split data into training and validation sets using scikit-learn
    if samples_for_train_val_split:
        train_samples, val_samples = train_test_split(
            samples_for_train_val_split,
            test_size=current_val_split_size,
            random_state=current_random_state,
            shuffle=True,  # Shuffle data before splitting
        )
        print(
            f"Total number of samples for train/val split: {len(samples_for_train_val_split)}"
        )
        print(f"Number of training samples: {len(train_samples)}")
        print(f"Number of validation samples: {len(val_samples)}")

        if not train_samples:
            print(
                "WARNING: Training set is empty after split. Check data and val_split_size."
            )
    else:
        print("WARNING: No samples provided for training/validation split.")
        train_samples, val_samples = [], []

    train_dataset = DrivingDataset(
        train_samples, images_dir, transform=data_transforms["train"]
    )

    # Initialize val_dataset only if there is validation data
    val_dataset = None
    if val_samples:
        val_dataset = DrivingDataset(
            val_samples, images_dir, transform=data_transforms["val"]
        )

    train_loader = None
    if train_dataset and len(train_dataset) > 0:
        train_loader = DataLoader(
            train_dataset,
            batch_size=current_batch_size,
            shuffle=True,
            num_workers=current_num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_fn_skip_broken,
        )

    val_loader = None
    if (
        val_dataset and len(val_dataset) > 0
    ):  # Check if val_dataset exists and is not empty
        val_loader = DataLoader(
            val_dataset,
            batch_size=current_batch_size,
            shuffle=False,  # Do not shuffle validation set
            num_workers=current_num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_fn_skip_broken,
        )
    elif not val_samples and samples_for_train_val_split:
        print(
            "WARNING: No validation samples after split, or val_split_size is 0 or 1."
        )

    # Test DataLoader
    test_loader = None
    if dedicated_test_samples:
        print(f"Number of dedicated test samples: {len(dedicated_test_samples)}")
        test_dataset = DrivingDataset(
            dedicated_test_samples,
            images_dir,
            transform=data_transforms["val"],  # Use 'val' transform for test
        )
        if len(test_dataset) > 0:
            test_loader = DataLoader(
                test_dataset,
                batch_size=current_batch_size,
                shuffle=False,
                num_workers=current_num_workers,
                pin_memory=torch.cuda.is_available(),
                collate_fn=collate_fn_skip_broken,
            )
        else:
            print(
                "WARNING: Test dataset is empty although dedicated_test_samples were provided."
            )
    else:
        print("INFO: No dedicated test samples provided.")

    return train_loader, val_loader, test_loader


# New main function to orchestrate data loading using YAML
def create_dataloaders_from_yaml(
    yaml_config_path,
    batch_size=None,
    val_split_size=None,
    random_state=None,
    num_workers=None,
):
    """
    Loads data configuration from YAML, parses annotations, splits data,
    and creates DataLoaders.
    """
    data_cfg = load_yaml_config(yaml_config_path)
    if not data_cfg:
        return None, None, None  # Failed to load YAML

    annotations_filepath = data_cfg.get("annotations_file")
    images_dir = data_cfg.get("images_dir")

    if not annotations_filepath or not images_dir:
        print("ERROR: 'annotations_file' or 'images_dir' not found in YAML config.")
        return None, None, None

    all_parsed_samples = parse_annotations_file(annotations_filepath)
    if not all_parsed_samples:
        print(f"ERROR: No samples parsed from '{annotations_filepath}'.")
        return None, None, None

    # Apply indices_to_skip
    indices_to_skip_yaml = data_cfg.get("indices_to_skip", [])
    if not isinstance(indices_to_skip_yaml, list):
        print(
            f"WARNING: 'indices_to_skip' in YAML is not a list. Found: {indices_to_skip_yaml}. Skipping this filter."
        )
        indices_to_skip_set = set()
    else:
        indices_to_skip_set = set(indices_to_skip_yaml)

    test_set_idx_start = data_cfg.get("test_set_idx_start")
    test_set_idx_end = data_cfg.get("test_set_idx_end")

    dedicated_test_samples = []
    samples_for_train_val_split = []

    for i, sample in enumerate(all_parsed_samples):
        if i in indices_to_skip_set:
            continue

        if i >= test_set_idx_start and i <= test_set_idx_end:
            dedicated_test_samples.append(sample)
        else:
            samples_for_train_val_split.append(sample)

    data_transforms = get_data_transforms()

    return get_dataloaders(
        samples_for_train_val_split,
        dedicated_test_samples,
        images_dir,
        data_transforms,
        batch_size=batch_size,
        val_split_size=val_split_size,
        random_state=random_state,
        num_workers=num_workers,
    )
