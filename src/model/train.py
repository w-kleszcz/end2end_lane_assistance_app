import os
import keras  # Keras 3.0

# Imports from local modules
import argparse
import config
import matplotlib.pyplot as plt
from data_loader import create_dataloaders_from_yaml # Updated import
from model import build_pilotnet_model
from tensorflow.keras.models import load_model


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
            print(
                "Please set the environment variable KERAS_BACKEND='torch' before running the script."
            )
            return False
        return True
    except Exception as e:
        print(f"An error occurred while setting the Keras backend: {e}")
        return False


def main():

    parser = argparse.ArgumentParser(description="Train or test the PilotNet model.")
    parser.add_argument(
        "--mode",
        choices=["train", "test"],
        default="train",
        help="Mode: 'train' to train and test, or 'test' to only test an existing model.",
    )
    parser.add_argument(
        "--data_config",
        type=str,
        required=True,
        help="Path to the data YAML configuration file (e.g., new_data/data.yaml).",
    )
    parser.add_argument(
        "--plot_save_path",
        type=str,
        default="steering_angle_evaluation.png",
        help="Path to save the evaluation plot (e.g., plots/evaluation.png).",
    )
    args = parser.parse_args()

    pilotnet_model = None

    # --- 1. Data Preparation ---
    print(f"Loading and preparing data using YAML config: {args.data_config}...")
    train_loader, val_loader, test_loader = create_dataloaders_from_yaml(args.data_config)

    if not train_loader and not val_loader and not test_loader:
        print("ERROR: Failed to create any DataLoaders from YAML. Aborting.")
        return

    if args.mode == "train":
        if not train_loader and not val_loader: # If both are None, specific for training
            print("ERROR: Failed to create training or validation DataLoaders. Aborting training.")
            return
        if train_loader is None:
            print("ERROR: Failed to create train_loader. Aborting.")
            return
        if not train_loader and not val_loader:  # If both are None
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
            loss="mse",  # Mean Squared Error for regression
        )
        pilotnet_model.summary()

        # --- 4. Model Training ---
        print(f"\nStarting training for {config.NUM_EPOCHS} epochs...")
        try:
            history = pilotnet_model.fit(
                train_loader,
                epochs=config.NUM_EPOCHS,
                validation_data=(
                    val_loader if val_loader and len(val_loader) > 0 else None
                ),  # Pass only if val_loader exists and is not empty
                verbose=1,  # 0 = silent, 1 = progress bar, 2 = one line per epoch
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
    elif args.mode == "test":
        pilotnet_model = load_model(config.MODEL_SAVE_PATH)

    # --- 6. Model Evaluation on Test Dataset ---
    if test_loader and len(test_loader) > 0:
        print("\nEvaluating model on the test set...")
        preds = []
        truths = []

        for images, labels in test_loader:
            outputs = pilotnet_model.predict(images, verbose=0)
            preds.extend(outputs.flatten())
            truths.extend(labels.flatten())

        # Plot predictions vs actual steering angles
        plt.figure(figsize=(10, 6))
        plt.plot(truths, label="Actual Steering Angles", alpha=0.7)
        plt.plot(preds, label="Predicted Steering Angles", alpha=0.7)
        plt.title("Steering Angle Prediction on Test Set")
        plt.xlabel("Sample Index in Test Set")
        plt.ylabel("Steering Angle")
        plt.legend()
        plt.grid(True)
        
        # Ensure the directory for saving the plot exists
        plot_dir = os.path.dirname(args.plot_save_path)
        if plot_dir and not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
            print(f"Created directory for plot: {plot_dir}")

        plt.savefig(args.plot_save_path)
        print(f"Evaluation plot saved to {args.plot_save_path}")
        # plt.show() # Keep this commented out or remove if interactive display is not needed

    elif args.mode == "test":
        print("WARNING: Test loader is not available or empty. Cannot perform evaluation on test set.")
    else: # train mode but no test_loader
        print("INFO: No test_loader available to perform evaluation after training.")

if __name__ == "__main__":
    # e.g., in the terminal: export KERAS_BACKEND="torch" && python train.py
    main()
