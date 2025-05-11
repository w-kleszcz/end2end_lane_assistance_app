# data_loader.py
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split # We use scikit-learn

# Import configuration
import config

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
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"WARNING: Image {img_path} not found, skipping sample.")
            return None, None # Will be handled by collate_fn
        except Exception as e:
            print(f"WARNING: Error loading image {img_path}: {e}, skipping sample.")
            return None, None

        if self.transform:
            image = self.transform(image)
        
        if image is None: # Additional safeguard
             return None, None

        return image, torch.tensor([steering_angle], dtype=torch.float32)

# Definition of image transformations
def get_data_transforms():
    """Returns a dictionary of transformations for training and validation sets."""
    return {
        'train': transforms.Compose([
            transforms.Resize((config.IMG_HEIGHT, config.IMG_WIDTH)),
            transforms.ToTensor(),  # Scales to [0.0, 1.0] and changes to C x H x W
            transforms.Lambda(lambda x: x - 0.5),  # Scales to [-0.5, 0.5]
            # Augmentations for the training set can be added here, e.g.:
            # transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]),
        'val': transforms.Compose([
            transforms.Resize((config.IMG_HEIGHT, config.IMG_WIDTH)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x - 0.5), # Scales to [-0.5, 0.5]
        ]),
    }

# collate_fn function to skip broken samples
def collate_fn_skip_broken(batch):
    """Filters out samples that returned None from Dataset.__getitem__."""
    # Remove (None, None) from the batch
    batch = list(filter(lambda x: x[0] is not None and x[1] is not None, batch))
    if not batch: # If the entire batch is empty after filtering
        return torch.tensor([]), torch.tensor([])
    return torch.utils.data.dataloader.default_collate(batch)

# Function to parse the annotations file
def parse_annotations_file(annotations_filepath):
    all_samples = []
    if not os.path.exists(annotations_filepath):
        print(f"ERROR: Annotations file '{annotations_filepath}' not found.")
        return all_samples

    with open(annotations_filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue # Skip empty lines

            parts = line.split(' ', 1) # Split only on the first space
            if len(parts) < 2:
                print(f"WARNING: Line {line_num} in '{annotations_filepath}' has incorrect format: '{line}', skipping.")
                continue
            
            img_name = parts[0]
            try:
                # Steering angle is the first value before the comma
                steering_angle_str = parts[1].split(',')[0]
                steering_angle = float(steering_angle_str)
                all_samples.append((img_name, steering_angle))
            except ValueError:
                print(f"WARNING: Line {line_num} in '{annotations_filepath}', cannot convert angle to float: '{parts[1]}', skipping.")
            except IndexError:
                 print(f"WARNING: Line {line_num} in '{annotations_filepath}', incorrect format after comma: '{parts[1]}', skipping.")
    return all_samples

def split_all_samples(all_samples, per_cent_test_set):
    """Splits the dataset into training and validation sets."""
    # Calculate the index for the split
    # Select every second sample starting from index 1 for test set
    test_samples = all_samples[1::2]  # Take every second sample starting from index 1
    all_samples = all_samples[::2]  # Take every second sample starting from index 0

    return all_samples, test_samples

# Main function to create DataLoaders
def get_dataloaders(all_samples, test_samples):
    """Creates and returns DataLoaders for training and validation sets."""

    if not all_samples:
        print("ERROR: No valid samples found in the annotations file. Cannot create DataLoaders.")
        return None, None # Return None if no data

    # Split data into training and validation sets using scikit-learn
    train_samples, val_samples = train_test_split(
        all_samples,
        test_size=config.VAL_SPLIT_SIZE,
        random_state=config.RANDOM_STATE,
        shuffle=True # Shuffle data before splitting
    )

    print(f"Total number of samples: {len(all_samples)}")
    print(f"Number of training samples: {len(train_samples)}")
    print(f"Number of validation samples: {len(val_samples)}")

    if not train_samples:
        print("ERROR: Training set is empty after split. Check data and val_split_size.")
        # return None, None # Can return None if there is no training data

    data_transforms = get_data_transforms()

    train_dataset = DrivingDataset(train_samples, config.IMAGES_DIR, transform=data_transforms['train'])
    
    # Initialize val_dataset only if there is validation data
    val_dataset = None
    if val_samples:
        val_dataset = DrivingDataset(val_samples, config.IMAGES_DIR, transform=data_transforms['val'])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(), # Use pin_memory if GPU is available
        collate_fn=collate_fn_skip_broken
    )
    
    val_loader = None
    if val_dataset and len(val_dataset) > 0: # Check if val_dataset exists and is not empty
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False, # Do not shuffle validation set
            num_workers=config.NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_fn_skip_broken
        )
    elif not val_samples:
        print("WARNING: No validation samples after split.")
    
    test_dataset = DrivingDataset(test_samples, config.IMAGES_DIR, transform=data_transforms['train'])
    test_loader = None
    if len(test_samples) > 0:
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False, # Do not shuffle validation set
            num_workers=config.NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_fn_skip_broken
        )
    else:
        print("WARNING: No test samples after split.")

    return train_loader, val_loader, test_loader