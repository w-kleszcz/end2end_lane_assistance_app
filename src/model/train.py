import os
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow # For core mlflow functions like mlflow.set_experiment, mlflow.start_run, etc.
from mlflow import pytorch as mlflow_pytorch_api # Using an alias for clarity

# Imports from local modules
import argparse
import config
import matplotlib.pyplot as plt
import numpy as np # For test MSE calculation
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
    parser.add_argument(
        "--mlflow_experiment_name",
        type=str,
        default=config.MLFLOW_DEFAULT_EXPERIMENT_NAME,
        help="Name of the MLflow experiment.",
    )
    parser.add_argument(
        "--mlflow_run_name",
        type=str,
        default=None,
        help="Name of the MLflow run (optional).",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- MLflow Setup ---
    mlflow.set_experiment(args.mlflow_experiment_name)

    with mlflow.start_run(run_name=args.mlflow_run_name) as run:
        print(f"MLflow Run ID: {run.info.run_id}")
        mlflow.log_param("run_mode", args.mode)
        mlflow.log_param("data_config_path", args.data_config)
        mlflow.log_param("plot_save_path_arg", args.plot_save_path)
        mlflow.log_param("device", str(device))

        # Log relevant config parameters that influence the run
        params_to_log_from_config = {
            "IMG_CHANNELS": config.IMG_CHANNELS,
            "IMG_HEIGHT": config.IMG_HEIGHT,
            "IMG_WIDTH": config.IMG_WIDTH,
            "MODEL_SAVE_PATH_config": config.MODEL_SAVE_PATH,
            # These might be overridden by data_loader logic or future CLI args
            "BATCH_SIZE_config": config.BATCH_SIZE,
            "NUM_EPOCHS_config": config.NUM_EPOCHS,
            "LEARNING_RATE_config": config.LEARNING_RATE,
            "VAL_SPLIT_SIZE_config": config.VAL_SPLIT_SIZE,
            "RANDOM_STATE_config": config.RANDOM_STATE,
            "LAMBDA_SMOOTHNESS_config": config.LAMBDA_SMOOTHNESS,
            "NUM_WORKERS_config": config.NUM_WORKERS,
        }
        mlflow.log_params(params_to_log_from_config)

        pilotnet_model = None

        # --- 1. Data Preparation ---
        print(f"Loading and preparing data using YAML config: {args.data_config}...")
        # create_dataloaders_from_yaml uses defaults from config if specific args not passed
        train_loader, val_loader, test_loader = create_dataloaders_from_yaml(
            args.data_config
            # batch_size, val_split_size, random_state, num_workers are taken from config by default
        )

        if not train_loader and not val_loader and not test_loader:
            print("ERROR: Failed to create any DataLoaders from YAML. Aborting.")
            mlflow.set_tag("data_loading_status", "failed_all_loaders")
            return

        if args.mode == "train":
            if not train_loader: # train_loader is essential for training
                print(
                    "ERROR: Failed to create training DataLoader. Aborting training."
                )
                mlflow.set_tag("data_loading_status", "failed_train_loader")
                return
            if len(train_loader) == 0:
                print("ERROR: train_loader is empty (contains no batches). Aborting.")
                mlflow.set_tag("data_loading_status", "train_loader_empty")
                return

            mlflow.log_param("actual_batch_size", train_loader.batch_size)
            mlflow.log_param("actual_num_workers", train_loader.num_workers)
            # VAL_SPLIT_SIZE and RANDOM_STATE are used by create_dataloaders_from_yaml from config

            # --- 2. Model Building ---
            print("Building the PilotNet model...")
            pilotnet_model = build_pilotnet_model(input_channels=config.IMG_CHANNELS)
            pilotnet_model.to(device)
            print("Model Summary:\n", pilotnet_model)

            # --- 3. Model Compilation ---
            print("Setting up optimizer and loss function...")
            optimizer = optim.Adam(pilotnet_model.parameters(), lr=config.LEARNING_RATE)
            criterion_mse = nn.MSELoss()
            mlflow.log_param("optimizer", "Adam")
            mlflow.log_param("learning_rate_actual", config.LEARNING_RATE)
            mlflow.log_param("loss_function", "MSELoss")
            mlflow.log_param("lambda_smoothness_actual", config.LAMBDA_SMOOTHNESS)

            # --- 4. Model Training ---
            print(f"\nStarting training for {config.NUM_EPOCHS} epochs...")
            mlflow.log_param("num_epochs_actual", config.NUM_EPOCHS)
            best_val_loss = float("inf")

            try:
                for epoch in range(config.NUM_EPOCHS):
                    pilotnet_model.train()
                    running_loss = 0.0
                    for i, (inputs, labels) in enumerate(train_loader):
                        inputs, labels = inputs.to(device), labels.to(device)
                        optimizer.zero_grad()
                        outputs = pilotnet_model(inputs)
                        loss_mse = criterion_mse(outputs, labels)
                        loss_smoothness = torch.tensor(0.0, device=device)
                        if outputs.size(0) > 1:
                            smoothness_diff = outputs[1:] - outputs[:-1]
                            loss_smoothness = torch.mean(smoothness_diff**2)
                        total_loss = loss_mse + config.LAMBDA_SMOOTHNESS * loss_smoothness
                        total_loss.backward()
                        optimizer.step()
                        running_loss += total_loss.item()

                    avg_train_loss = running_loss / len(train_loader)
                    mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
                    log_msg = f"Epoch {epoch+1}/{config.NUM_EPOCHS} - loss: {avg_train_loss:.4f}"

                    if val_loader and len(val_loader) > 0:
                        pilotnet_model.eval()
                        val_running_loss = 0.0
                        with torch.no_grad():
                            for inputs_val, labels_val in val_loader:
                                inputs_val, labels_val = inputs_val.to(device), labels_val.to(device)
                                outputs_val = pilotnet_model(inputs_val)
                                loss_mse_val = criterion_mse(outputs_val, labels_val)
                                loss_smoothness_val = torch.tensor(0.0, device=device)
                                if outputs_val.size(0) > 1:
                                    smoothness_diff_val = outputs_val[1:] - outputs_val[:-1]
                                    loss_smoothness_val = torch.mean(smoothness_diff_val**2)
                                total_loss_val = loss_mse_val + config.LAMBDA_SMOOTHNESS * loss_smoothness_val
                                val_running_loss += total_loss_val.item()

                        avg_val_loss = val_running_loss / len(val_loader)
                        mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
                        log_msg += f" - val_loss: {avg_val_loss:.4f}"

                        if avg_val_loss < best_val_loss:
                            best_val_loss = avg_val_loss
                            torch.save(pilotnet_model.state_dict(), config.MODEL_SAVE_PATH)
                            log_msg += " (New best model saved)"
                    else:
                        if epoch == config.NUM_EPOCHS - 1: # No validation, save last model
                            torch.save(pilotnet_model.state_dict(), config.MODEL_SAVE_PATH)
                            log_msg += " (Model saved at end of training)"
                    print(log_msg)

                print("\nTraining completed.")
                mlflow.log_metric("best_val_loss", best_val_loss if best_val_loss != float('inf') else -1.0)

                if os.path.exists(config.MODEL_SAVE_PATH):
                    print(f"Model saved to: {config.MODEL_SAVE_PATH}")
                    mlflow.log_artifact(config.MODEL_SAVE_PATH, artifact_path="model_checkpoint")
                    mlflow_pytorch_api.log_model(pilotnet_model, "model")
                else:
                    print(f"Model was not saved to {config.MODEL_SAVE_PATH}")
                    mlflow.set_tag("model_saved", "false")

            except Exception as e:
                print(f"\nAn error occurred during training or saving the model: {e}")
                mlflow.set_tag("training_status", "error")
                mlflow.set_tag("error_message", str(e))
                import traceback
                traceback.print_exc()
                return

            mlflow.set_tag("training_status", "completed")

        elif args.mode == "test":
            mlflow.set_tag("run_type", "test_only")
            if not os.path.exists(config.MODEL_SAVE_PATH):
                print(f"ERROR: Model file not found at {config.MODEL_SAVE_PATH} for testing.")
                mlflow.set_tag("test_status", "model_not_found")
                return
            pilotnet_model = build_pilotnet_model(input_channels=config.IMG_CHANNELS)
            pilotnet_model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
            pilotnet_model.to(device)
            print(f"Loaded model from {config.MODEL_SAVE_PATH} for testing.")
            mlflow.log_param("tested_model_path", config.MODEL_SAVE_PATH)

        # --- 6. Model Evaluation on Test Dataset ---
        if test_loader and len(test_loader) > 0 and pilotnet_model: # Ensure model is available
            print("\nEvaluating model on the test set...")
            preds, truths = [], []
            pilotnet_model.eval()
            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(device)
                    outputs = pilotnet_model(images)
                    preds.extend(outputs.cpu().numpy().flatten())
                    truths.extend(labels.numpy().flatten())

            if preds and truths:
                test_mse = ((np.array(preds) - np.array(truths)) ** 2).mean()
                print(f"Test MSE: {test_mse:.4f}")
                mlflow.log_metric("test_mse", test_mse)

            plt.figure(figsize=(10, 6))
            plt.plot(truths, label="Actual Steering Angles", alpha=0.7)
            plt.plot(preds, label="Predicted Steering Angles", alpha=0.7)
            plt.title("Steering Angle Prediction on Test Set")
            plt.xlabel("Sample Index in Test Set")
            plt.ylabel("Steering Angle")
            plt.legend()
            plt.grid(True)

            plot_dir = os.path.dirname(args.plot_save_path)
            if plot_dir and not os.path.exists(plot_dir):
                os.makedirs(plot_dir)
            plt.savefig(args.plot_save_path)
            print(f"Evaluation plot saved to {args.plot_save_path}")
            mlflow.log_artifact(args.plot_save_path, artifact_path="evaluation_plots")
            plt.close()
            mlflow.set_tag("test_evaluation", "performed")
        elif args.mode == "test": # Test mode but no test_loader or model
            print("WARNING: Test loader is not available or empty, or model not loaded. Cannot perform evaluation.")
            mlflow.set_tag("test_evaluation", "skipped_no_loader_or_model")
        else: # Train mode but no test_loader
            print("INFO: No test_loader available to perform evaluation after training.")
            mlflow.set_tag("test_evaluation", "skipped_no_loader_train_mode")


if __name__ == "__main__":
    # e.g., in the terminal: export KERAS_BACKEND="torch" && python train.py
    main()
