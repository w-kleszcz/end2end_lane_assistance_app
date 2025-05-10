import os
import keras # Keras 3.0

# Imports from local modules
import config
from data_loader import get_dataloaders
from model import build_pilotnet_model


def set_keras_backend():
    """Sets the Keras backend and verifies it."""
    try:
        if os.environ.get("KERAS_BACKEND") != "torch":
            print("Setting Keras backend to 'torch'...")
            os.environ["KERAS_BACKEND"] = "torch"
        
        import keras as current_keras_instance
        print(f"Keras backend in use: {current_keras_instance.backend.backend()}")
        if current_keras_instance.backend.backend() != "torch":
            print("ERROR: Failed to set Keras backend to 'torch'.")
            print("Please set the environment variable KERAS_BACKEND='torch' before running the script.")
            return False
        return True
    except Exception as e:
        print(f"An error occurred while setting the Keras backend: {e}")
        return False


def main():

 # --- 1. Data Preparation ---
    print("Loading and preparing data...")
    train_loader, val_loader = get_dataloaders()

    if train_loader is None:
        print("ERROR: Failed to create train_loader. Aborting.")
        return
    if not train_loader and not val_loader: # If both are None
        print("ERROR: Failed to create any DataLoaders. Aborting.")
        return
    
    # Check if train_loader has any batches
    if len(train_loader) == 0:
        print("ERROR: train_loader is empty (contains no batches). Aborting.")
        print("Check the annotations file, image paths, and BATCH_SIZE.")
        return

    # --- 2. Model Building ---
    print("Building the PilotNet model...")
    pilotnet_model = build_pilotnet_model(config.MODEL_INPUT_SHAPE)
    
    # --- 3. Model Compilation ---
    print("Compiling the model...")
    pilotnet_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss='mse' # Mean Squared Error for regression
    )
    pilotnet_model.summary()
    
    # --- 4. Model Training ---
    print(f"\nStarting training for {config.NUM_EPOCHS} epochs...")
    try:
        history = pilotnet_model.fit(
            train_loader,
            epochs=config.NUM_EPOCHS,
            validation_data=val_loader if val_loader and len(val_loader) > 0 else None, # Pass only if val_loader exists and is not empty
            verbose=1 # 0 = silent, 1 = progress bar, 2 = one line per epoch
        )
        print("\nTraining completed.")

        # --- 5. Model Saving ---
        print(f"Saving the model to: {config.MODEL_SAVE_PATH}...")
        pilotnet_model.save(config.MODEL_SAVE_PATH)
        print("Model saved.")

    except Exception as e:
        print(f"\nAn error occurred during training or saving the model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    # e.g., in the terminal: export KERAS_BACKEND="torch" && python train.py
    main()
