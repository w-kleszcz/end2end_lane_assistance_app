# --- Data Parameters ---
# ANNOTATIONS_FILE and IMAGES_DIR are now specified in the data YAML file.
# ANNOTATIONS_FILE = "data/raw/07012018/data.txt"
# IMAGES_DIR = "data/raw/07012018/data"

# Image dimensions (for PyTorch backend: Channels, Height, Width)
IMG_CHANNELS = 3
IMG_HEIGHT = 66
IMG_WIDTH = 200
# Input shape for Keras model with PyTorch backend
MODEL_INPUT_SHAPE = (IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH)

# --- Training Parameters ---
BATCH_SIZE = 32
NUM_EPOCHS = 10  # You can increase this for actual training
LEARNING_RATE = 1e-4
VAL_SPLIT_SIZE = 0.2  # 20% of data for the validation set
RANDOM_STATE = 42  # For ensuring reproducibility of data split

# --- Model Saving ---
MODEL_SAVE_PATH = "models/pilotnet_keras_torch_backend.keras"

# --- DataLoader Parameters ---
# Set to >0 for multi-threaded loading, e.g., 2 or 4.
# On Windows or in some environments (e.g., Jupyter), it might be necessary to set this to 0.
NUM_WORKERS = 0
