import os
import torch
import torch.nn as nn
import torch.optim as optim

# Imports from local modules
import argparse
import config
import matplotlib.pyplot as plt
from data_loader import create_dataloaders_from_yaml  # Updated import
from model import build_pilotnet_model  # This will now return a PyTorch model


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    pilotnet_model = None

    # --- 1. Data Preparation ---
    print(f"Loading and preparing data using YAML config: {args.data_config}...")
    train_loader, val_loader, test_loader = create_dataloaders_from_yaml(
        args.data_config
    )

    if not train_loader and not val_loader and not test_loader:
        print("ERROR: Failed to create any DataLoaders from YAML. Aborting.")
        return

    if args.mode == "train":
        if (
            not train_loader and not val_loader
        ):  # If both are None, specific for training
            print(
                "ERROR: Failed to create training or validation DataLoaders. Aborting training."
            )
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
        pilotnet_model = build_pilotnet_model(input_channels=config.IMG_CHANNELS)
        pilotnet_model.to(device)
        print("Model Summary:\n", pilotnet_model)

        # --- 3. Model Compilation ---
        print("Setting up optimizer and loss function...")
        optimizer = optim.Adam(pilotnet_model.parameters(), lr=config.LEARNING_RATE)
        criterion = nn.MSELoss()

        # --- 4. Model Training ---
        print(f"\nStarting training for {config.NUM_EPOCHS} epochs...")
        best_val_loss = float("inf")
        # For simplicity, train.py won't implement early stopping patience from GUI
        # but will save the best model based on val_loss if val_loader is present.

        try:
            for epoch in range(config.NUM_EPOCHS):
                pilotnet_model.train()
                running_loss = 0.0
                for i, (inputs, labels) in enumerate(train_loader):
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = pilotnet_model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

                avg_train_loss = running_loss / len(train_loader)
                log_msg = (
                    f"Epoch {epoch+1}/{config.NUM_EPOCHS} - loss: {avg_train_loss:.4f}"
                )

                if val_loader and len(val_loader) > 0:
                    pilotnet_model.eval()
                    val_running_loss = 0.0
                    with torch.no_grad():
                        for inputs_val, labels_val in val_loader:
                            inputs_val, labels_val = inputs_val.to(
                                device
                            ), labels_val.to(device)
                            outputs_val = pilotnet_model(inputs_val)
                            loss_val = criterion(outputs_val, labels_val)
                            val_running_loss += loss_val.item()
                    avg_val_loss = val_running_loss / len(val_loader)
                    log_msg += f" - val_loss: {avg_val_loss:.4f}"

                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        torch.save(pilotnet_model.state_dict(), config.MODEL_SAVE_PATH)
                        log_msg += " (New best model saved)"
                else:  # No validation loader, save at the end
                    if epoch == config.NUM_EPOCHS - 1:
                        torch.save(pilotnet_model.state_dict(), config.MODEL_SAVE_PATH)
                        log_msg += " (Model saved at end of training)"

                print(log_msg)

            print("\nTraining completed.")
            if not (
                val_loader and len(val_loader) > 0
            ):  # If no val_loader, best_val_loss is inf
                if os.path.exists(config.MODEL_SAVE_PATH):
                    print(f"Model saved to: {config.MODEL_SAVE_PATH}")
            elif best_val_loss != float("inf"):
                print(
                    f"Best model (val_loss: {best_val_loss:.4f}) saved to: {config.MODEL_SAVE_PATH}"
                )

        except Exception as e:
            print(f"\nAn error occurred during training or saving the model: {e}")
            import traceback

            traceback.print_exc()
            return  # Exit if training failed

    elif args.mode == "test":
        if not os.path.exists(config.MODEL_SAVE_PATH):
            print(
                f"ERROR: Model file not found at {config.MODEL_SAVE_PATH} for testing."
            )
            return
        pilotnet_model = build_pilotnet_model(input_channels=config.IMG_CHANNELS)
        pilotnet_model.load_state_dict(
            torch.load(config.MODEL_SAVE_PATH, map_location=device)
        )
        pilotnet_model.to(device)
        print(f"Loaded model from {config.MODEL_SAVE_PATH} for testing.")

    # --- 6. Model Evaluation on Test Dataset ---
    if test_loader and len(test_loader) > 0:
        print("\nEvaluating model on the test set...")
        preds = []
        truths = []

        pilotnet_model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                outputs = pilotnet_model(images)
                preds.extend(outputs.cpu().numpy().flatten())
                truths.extend(labels.numpy().flatten())

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
        print(
            "WARNING: Test loader is not available or empty. Cannot perform evaluation on test set."
        )
    else:  # train mode but no test_loader
        print("INFO: No test_loader available to perform evaluation after training.")


if __name__ == "__main__":
    # e.g., in the terminal: export KERAS_BACKEND="torch" && python train.py
    main()
