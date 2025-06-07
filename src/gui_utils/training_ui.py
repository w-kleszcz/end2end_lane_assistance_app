import os
import queue
import threading
import dearpygui.dearpygui as dpg
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import mlflow
from mlflow import pytorch as mlflow_pytorch_api
import numpy as np

from model.data_loader import create_dataloaders_from_yaml
from model.model import build_pilotnet_model
import model.config as global_train_config 

UI_INPUT_WIDTH_LONG = 550  # Define it here or pass as argument

# --- Globals for Model Training UI within this module ---
training_log_queue = queue.Queue()

# UI Element Tags (scoped to this module if not needed outside)
TRAINING_LOG_TAG = "training_ui::training_log_text_area"
TRAINING_STATUS_TAG = "training_ui::training_status_indicator"
TRAIN_BUTTON_TAG = "training_ui::train_model_button_tag"
DATA_CONFIG_PATH_INPUT_TAG = "training_ui::data_config_path_input_tag"
PLOT_SAVE_PATH_INPUT_TAG = "training_ui::plot_save_path_input_tag"
MODEL_SAVE_PATH_INPUT_TAG = "training_ui::model_save_path_input_tag"
BATCH_SIZE_INPUT_TAG = "training_ui::batch_size_input_tag"
NUM_EPOCHS_INPUT_TAG = "training_ui::num_epochs_input_tag"
LEARNING_RATE_INPUT_TAG = "training_ui::learning_rate_input_tag"
VAL_SPLIT_SIZE_INPUT_TAG = "training_ui::val_split_size_input_tag"
RANDOM_STATE_INPUT_TAG = "training_ui::random_state_input_tag"
NUM_WORKERS_INPUT_TAG = "training_ui::num_workers_input_tag"
EARLY_STOPPING_PATIENCE_INPUT_TAG = "training_ui::early_stopping_patience_input_tag"
LAMBDA_SMOOTHNESS_INPUT_TAG = "training_ui::lambda_smoothness_input_tag"
MLFLOW_ENABLE_CHECKBOX_TAG = "training_ui::mlflow_enable_checkbox_tag"
MLFLOW_EXPERIMENT_NAME_INPUT_TAG = "training_ui::mlflow_experiment_name_input_tag"
MLFLOW_RUN_NAME_INPUT_TAG = "training_ui::mlflow_run_name_input_tag"

def log_to_gui(message):
    """Puts a message into the training log queue."""
    training_log_queue.put(message)


def format_log_message(epoch, logs, total_epochs):
    """Formats the log message for a PyTorch epoch."""
    loss_val = logs.get("loss")
    val_loss = logs.get("val_loss")

    loss_str = f"{loss_val:.4f}" if loss_val is not None else "N/A"
    val_loss_str = f"{val_loss:.4f}" if val_loss is not None else "N/A"

    return (
        f"Epoch {epoch+1}/{total_epochs} - "
        f"loss: {loss_str} - "
        f"val_loss: {val_loss_str}\n"
    )


def run_training_thread(
    data_config_path,
    plot_save_path,
    model_save_path,
    batch_size,
    num_epochs,
    learning_rate,
    val_split_size,
    random_state,
    num_workers,
    early_stopping_patience,
    lambda_smoothness,
    mlflow_enabled,
    mlflow_experiment_name,
    mlflow_run_name,
):
    """The main logic for training, adapted from train.py."""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log_to_gui(f"Using device: {device}\n")

        log_to_gui("--- Training Process Started ---\n")
        log_to_gui(f"Using data config: {data_config_path}\n")
        log_to_gui(f"Evaluation plot will be saved to: {plot_save_path}\n")
        log_to_gui(f"Model will be saved to: {model_save_path}\n")
        log_to_gui("Training Parameters:\n")
        log_to_gui(f"  Batch Size: {batch_size}\n")
        log_to_gui(f"  Num Epochs: {num_epochs}\n")
        log_to_gui(f"  Learning Rate: {learning_rate}\n")
        log_to_gui(f"  Validation Split Size: {val_split_size}\n")
        log_to_gui(f"  Random State: {random_state}\n")
        log_to_gui(f"  Num Workers: {num_workers}\n")
        log_to_gui(f"  Early Stopping Patience: {early_stopping_patience}\n\n")
        log_to_gui(f"  Lambda Smoothness: {lambda_smoothness}\n")
        log_to_gui(f"  MLflow Enabled: {mlflow_enabled}\n")
        if mlflow_enabled:
            log_to_gui(f"  MLflow Experiment: {mlflow_experiment_name}\n")
            log_to_gui(f"  MLflow Run Name: {mlflow_run_name if mlflow_run_name else 'Default (auto-generated)'}\n\n")
        else:
            log_to_gui("\n")

        if mlflow_enabled:
            mlflow.set_experiment(mlflow_experiment_name)
            with mlflow.start_run(run_name=mlflow_run_name if mlflow_run_name else None) as run:
                log_to_gui(f"MLflow Run ID: {run.info.run_id}\n")
                _perform_training_steps(
                    data_config_path, plot_save_path, model_save_path, batch_size, num_epochs,
                    learning_rate, val_split_size, random_state, num_workers,
                    early_stopping_patience, lambda_smoothness, device, mlflow_enabled=True
                )
        else:
            _perform_training_steps(
                data_config_path, plot_save_path, model_save_path, batch_size, num_epochs,
                learning_rate, val_split_size, random_state, num_workers,
                early_stopping_patience, lambda_smoothness, device, mlflow_enabled=False
            )

    except Exception as e:
        log_to_gui(f"\n--- ERROR during training process ---\n{str(e)}\n")
        import traceback
        log_to_gui(traceback.format_exc() + "\n")
        dpg.set_value(TRAINING_STATUS_TAG, "Status: Error")
        if mlflow_enabled and mlflow.active_run():
            mlflow.set_tag("training_status", "error_outer_scope")
            mlflow.set_tag("error_message", str(e))
            mlflow.end_run(status="FAILED")
    finally:
        dpg.configure_item(TRAIN_BUTTON_TAG, enabled=True)


def _perform_training_steps(
    data_config_path, plot_save_path, model_save_path, batch_size, num_epochs,
    learning_rate, val_split_size, random_state, num_workers,
    early_stopping_patience, lambda_smoothness, device, mlflow_enabled
):
    """Helper function containing the core training and evaluation logic."""
    
    if mlflow_enabled:
        params_to_log = {
            "data_config_path": data_config_path,
            "plot_save_path": plot_save_path,
            "model_save_path": model_save_path,
            "batch_size_gui": batch_size,
            "num_epochs_gui": num_epochs,
            "learning_rate_gui": learning_rate,
            "val_split_size_gui": val_split_size,
            "random_state_gui": random_state,
            "num_workers_gui": num_workers,
            "early_stopping_patience_gui": early_stopping_patience,
            "lambda_smoothness_gui": lambda_smoothness,
            "IMG_CHANNELS_config": global_train_config.IMG_CHANNELS,
            "IMG_HEIGHT_config": global_train_config.IMG_HEIGHT,
            "IMG_WIDTH_config": global_train_config.IMG_WIDTH,
            "device": str(device),
        }
        mlflow.log_params(params_to_log)

    try:
        train_loader, val_loader, test_loader = create_dataloaders_from_yaml(
            data_config_path,
            batch_size=batch_size,
            val_split_size=val_split_size,
            random_state=random_state,
            num_workers=num_workers,
        )

        if not train_loader:
            log_to_gui("ERROR: Failed to create training DataLoader. Aborting.\n")
            dpg.set_value(TRAINING_STATUS_TAG, "Error: Training data missing")
            return
        if len(train_loader) == 0:
            log_to_gui(
                "ERROR: train_loader is empty. Check annotations, paths, BATCH_SIZE.\n"
            )
            dpg.set_value(TRAINING_STATUS_TAG, "Error: Training data empty")
            return
        log_to_gui(f"Data loaded successfully. Train batches: {len(train_loader)}, Val batches: {len(val_loader) if val_loader else 0}, Test batches: {len(test_loader) if test_loader else 0}\n")
        if mlflow_enabled:
            mlflow.log_param("actual_train_loader_batch_size", train_loader.batch_size)
            mlflow.log_param("actual_train_loader_num_workers", train_loader.num_workers)
            mlflow.set_tag("data_loading_status", "success")
            
        pilotnet_model = build_pilotnet_model(
            input_channels=global_train_config.IMG_CHANNELS
        )
        pilotnet_model.to(device)
        log_to_gui("Model built.\n" + str(pilotnet_model) + "\n")

        optimizer = optim.Adam(pilotnet_model.parameters(), lr=learning_rate)
        criterion_mse = nn.MSELoss()
        log_to_gui("Optimizer and loss function set up.\n")

        log_to_gui(f"Starting training for {num_epochs} epochs...\n")
        dpg.set_value(TRAINING_STATUS_TAG, "Status: Training...")

        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

        best_val_loss = float("inf")
        epochs_no_improve = 0
        stopped_epoch_num = 0

        for epoch in range(num_epochs):
            pilotnet_model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = pilotnet_model(inputs)
                loss_mse = criterion_mse(outputs, labels)
                loss_smoothness = torch.tensor(0.0, device=device)
                if outputs.size(0) > 1 and lambda_smoothness > 0:
                    smoothness_diff = outputs[1:] - outputs[:-1]
                    loss_smoothness = torch.mean(smoothness_diff**2)
                
                total_loss = loss_mse + lambda_smoothness * loss_smoothness
                total_loss.backward()
                optimizer.step()
                running_loss += total_loss.item()

            avg_train_loss = running_loss / len(train_loader)
            current_logs = {"loss": avg_train_loss}
            if mlflow_enabled:
                mlflow.log_metric("train_loss", avg_train_loss, step=epoch)

            avg_val_loss = None
            if val_loader and len(val_loader) > 0:
                pilotnet_model.eval()
                val_running_loss = 0.0
                with torch.no_grad():
                    for inputs_val, labels_val in val_loader:
                        inputs_val, labels_val = inputs_val.to(device), labels_val.to(
                            device
                        )
                        outputs_val = pilotnet_model(inputs_val)
                        
                        loss_mse_val = criterion_mse(outputs_val, labels_val)

                        loss_smoothness_val = torch.tensor(0.0, device=device)
                        if outputs_val.size(0) > 1 and lambda_smoothness > 0:
                            smoothness_diff_val = outputs_val[1:] - outputs_val[:-1]
                            loss_smoothness_val = torch.mean(smoothness_diff_val**2)
                        
                        total_loss_val = loss_mse_val + lambda_smoothness * loss_smoothness_val
                        val_running_loss += total_loss_val.item()

                avg_val_loss = val_running_loss / len(val_loader)
                current_logs["val_loss"] = avg_val_loss
                if mlflow_enabled:
                    mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
                log_to_gui(format_log_message(epoch, current_logs, num_epochs))

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(pilotnet_model.state_dict(), model_save_path)
                    log_to_gui(
                        f"Epoch {epoch+1}: val_loss improved to {avg_val_loss:.4f}, model saved.\n"
                    )
                    if mlflow_enabled:
                        mlflow.set_tag(f"epoch_{epoch+1}_val_loss_improved", True)
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    log_to_gui(
                        f"Epoch {epoch+1}: val_loss ({avg_val_loss:.4f}) did not improve from {best_val_loss:.4f}.\n"
                    )
                    if mlflow_enabled:
                         mlflow.set_tag(f"epoch_{epoch+1}_val_loss_improved", False)
                if (
                    early_stopping_patience > 0
                    and epochs_no_improve >= early_stopping_patience
                ):
                    stopped_epoch_num = epoch + 1
                    log_to_gui(
                        f"Early stopping triggered after {epochs_no_improve} epochs without improvement.\n"
                    )
                    if mlflow_enabled:
                        mlflow.set_tag("early_stopping", "triggered")
                    break
            else:
                log_to_gui(format_log_message(epoch, current_logs, num_epochs))
                if epoch == num_epochs - 1:
                    torch.save(pilotnet_model.state_dict(), model_save_path)
                    log_to_gui(f"Model saved at end of training (no validation).\n")

        log_to_gui("Training completed.\n")
        if mlflow_enabled:
            mlflow.log_metric("final_best_val_loss", best_val_loss if best_val_loss != float('inf') else -1.0)
            if stopped_epoch_num > 0:
                mlflow.log_param("stopped_epoch", stopped_epoch_num)
            if not (val_loader and len(val_loader) > 0) and os.path.exists(model_save_path):
                 mlflow.set_tag("model_saved_reason", "end_of_training_no_validation")
            elif best_val_loss != float('inf') and os.path.exists(model_save_path):
                 mlflow.set_tag("model_saved_reason", "validation_improvement")
        
        if stopped_epoch_num > 0:
            log_to_gui(f"Early stopping was triggered at epoch {stopped_epoch_num}.\n")

        if os.path.exists(model_save_path):
            log_to_gui(f"Best model saved to: {model_save_path}\n")
            if mlflow_enabled:
                mlflow.log_artifact(model_save_path, artifact_path="model_checkpoint")
                mlflow_pytorch_api.log_model(pilotnet_model, "model")
        else:
            log_to_gui("No model was saved.\n")
            if mlflow_enabled: mlflow.set_tag("model_saved", "false")

        if test_loader and len(test_loader) > 0:
            log_to_gui("Evaluating model on the test set...\n")
            preds, truths = [], []
            eval_model = build_pilotnet_model(
                input_channels=global_train_config.IMG_CHANNELS
            )
            if os.path.exists(model_save_path):
                eval_model.load_state_dict(
                    torch.load(model_save_path, map_location=device)
                )
            else:  # Fallback to model in memory if save failed or no validation
                eval_model.load_state_dict(pilotnet_model.state_dict())

            eval_model.to(device)
            eval_model.eval()
            with torch.no_grad():
                for images, labels in test_loader:
                    outputs = eval_model(images.to(device))
                    preds.extend(outputs.cpu().numpy().flatten())
                    truths.extend(labels.numpy().flatten())

            plt.figure(figsize=(10, 6))
            plt.plot(truths, label="Actual Steering Angles", alpha=0.7)
            plt.plot(preds, label="Predicted Steering Angles", alpha=0.7)
            plt.title("Steering Angle Prediction on Test Set")
            plt.xlabel("Sample Index in Test Set")
            plt.ylabel("Steering Angle")
            plt.legend()
            plt.grid(True)

            plot_dir = os.path.dirname(plot_save_path)
            if plot_dir and not os.path.exists(plot_dir):
                os.makedirs(plot_dir)
            plt.savefig(plot_save_path)
            log_to_gui(f"Evaluation plot saved to {plot_save_path}\n")
            if mlflow_enabled:
                mlflow.log_artifact(plot_save_path, artifact_path="evaluation_plots")
            plt.close() # Close plot to free memory

            if preds and truths: # Calculate and log test MSE
                test_mse = ((np.array(preds) - np.array(truths)) ** 2).mean()
                log_to_gui(f"Test MSE: {test_mse:.4f}\n")
                if mlflow_enabled: mlflow.log_metric("test_mse", test_mse)
        else:
            log_to_gui("INFO: No test_loader available for evaluation.\n")
            if mlflow_enabled: mlflow.set_tag("test_evaluation_status", "skipped_no_loader")


        log_to_gui("--- Training Process Finished Successfully ---\n")
        dpg.set_value(TRAINING_STATUS_TAG, "Status: Completed")

        if mlflow_enabled: mlflow.set_tag("training_status", "completed")

    except Exception as e: # Catch errors within _perform_training_steps
        log_to_gui(f"\n--- ERROR during training steps ---\n{str(e)}\n")
        import traceback

        log_to_gui(traceback.format_exc() + "\n")
        dpg.set_value(TRAINING_STATUS_TAG, "Status: Error")
        if mlflow_enabled:
            mlflow.set_tag("training_status", "error_inner_scope")
            mlflow.set_tag("error_message", str(e))
            # The run will be ended by the outer context manager


def on_start_training_clicked():
    # Get values from DPG items
    data_config = dpg.get_value(DATA_CONFIG_PATH_INPUT_TAG)
    plot_path = dpg.get_value(PLOT_SAVE_PATH_INPUT_TAG)
    model_save_path = dpg.get_value(MODEL_SAVE_PATH_INPUT_TAG)
    batch_size = dpg.get_value(BATCH_SIZE_INPUT_TAG)
    num_epochs = dpg.get_value(NUM_EPOCHS_INPUT_TAG)
    learning_rate = dpg.get_value(LEARNING_RATE_INPUT_TAG)
    val_split_size = dpg.get_value(VAL_SPLIT_SIZE_INPUT_TAG)
    random_state = dpg.get_value(RANDOM_STATE_INPUT_TAG)
    num_workers = dpg.get_value(NUM_WORKERS_INPUT_TAG)
    early_stopping_patience = dpg.get_value(EARLY_STOPPING_PATIENCE_INPUT_TAG)
    lambda_smoothness = dpg.get_value(LAMBDA_SMOOTHNESS_INPUT_TAG)

    # Get MLflow UI values
    mlflow_enabled = dpg.get_value(MLFLOW_ENABLE_CHECKBOX_TAG)
    mlflow_experiment_name = dpg.get_value(MLFLOW_EXPERIMENT_NAME_INPUT_TAG)
    mlflow_run_name = dpg.get_value(MLFLOW_RUN_NAME_INPUT_TAG)

    # Basic Validations
    if not data_config or not os.path.isfile(data_config):
        log_to_gui("ERROR: Data Config YAML path is invalid or file does not exist.\n")
        dpg.set_value(TRAINING_STATUS_TAG, "Error: Invalid Data Config Path")
        return
    if mlflow_enabled and not mlflow_experiment_name:
        log_to_gui("ERROR: MLflow Experiment Name is required when MLflow is enabled.\n")
        dpg.set_value(TRAINING_STATUS_TAG, "Error: MLflow Experiment Name missing")
        return
    # Add more validations as in the original app.py for other fields...

    dpg.set_value(TRAINING_LOG_TAG, "")  # Clear previous logs
    dpg.set_value(TRAINING_STATUS_TAG, "Status: Starting...")
    dpg.configure_item(TRAIN_BUTTON_TAG, enabled=False)

    thread = threading.Thread(
        target=run_training_thread,
        args=(
            data_config,
            plot_path,
            model_save_path,
            batch_size,
            num_epochs,
            learning_rate,
            val_split_size,
            random_state,
            num_workers,
            early_stopping_patience,
            lambda_smoothness,
            mlflow_enabled,
            mlflow_experiment_name,
            mlflow_run_name,
        ),
        daemon=True,
    )
    thread.start()


def update_training_log_display():
    """Checks queue and updates the log display."""
    try:
        while not training_log_queue.empty():
            message = training_log_queue.get_nowait()
            current_log = dpg.get_value(TRAINING_LOG_TAG)
            dpg.set_value(TRAINING_LOG_TAG, current_log + message)
            # Consider adding auto-scrolling if DPG supports it easily for text areas
    except queue.Empty:
        pass
    except Exception as e:
        # Catch cases where the UI element might not exist yet or other DPG errors
        print(f"Error updating training log display: {e}")


def _select_file_callback(sender, app_data, user_data_tag):
    dpg.set_value(user_data_tag, app_data["file_path_name"])


def create_training_tab_content(parent_tab_id):
    with dpg.tab(label="Model Training", parent=parent_tab_id):
        dpg.add_text("Configure and start model training based on a data YAML file.")

        # File Dialogs (defined once, shown on button click)
        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            callback=_select_file_callback,
            user_data=DATA_CONFIG_PATH_INPUT_TAG,
            tag="training_ui::data_config_file_dialog",
            width=500,
            height=400,
        ):
            dpg.add_file_extension(".yaml", color=(255, 255, 0, 255))
            dpg.add_file_extension(".yml", color=(255, 255, 0, 255))
        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            callback=_select_file_callback,
            user_data=PLOT_SAVE_PATH_INPUT_TAG,
            tag="training_ui::plot_save_file_dialog",
            width=500,
            height=400,
            default_filename="steering_angle_evaluation.png",
        ):
            dpg.add_file_extension(".png", color=(0, 255, 0, 255))
            dpg.add_file_extension(".*")
        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            callback=_select_file_callback,
            user_data=MODEL_SAVE_PATH_INPUT_TAG,
            tag="training_ui::model_save_file_dialog",
            width=500,
            height=400,
            default_filename="pilotnet_pytorch.pth",
        ):
            dpg.add_file_extension(".pth", color=(0, 255, 255, 255))
            dpg.add_file_extension(".*")

        dpg.add_text("Data Configuration YAML:")
        with dpg.group(horizontal=True):
            dpg.add_button(
                label="Browse##DataConfig",
                callback=lambda: dpg.show_item("training_ui::data_config_file_dialog"),
            )
            dpg.add_input_text(
                tag=DATA_CONFIG_PATH_INPUT_TAG,
                width=UI_INPUT_WIDTH_LONG - 70,
                hint="e.g., new_data/data.yaml",
            )

        dpg.add_text("Evaluation Plot Save Path:")
        with dpg.group(horizontal=True):
            dpg.add_button(
                label="Browse##PlotSave",
                callback=lambda: dpg.show_item("training_ui::plot_save_file_dialog"),
            )
            dpg.add_input_text(
                tag=PLOT_SAVE_PATH_INPUT_TAG,
                width=UI_INPUT_WIDTH_LONG - 70,
                default_value="plots/steering_angle_evaluation.png",
                hint="e.g., plots/evaluation.png",
            )

        dpg.add_text("Model Save Path:")
        with dpg.group(horizontal=True):
            dpg.add_button(
                label="Browse##ModelSave",
                callback=lambda: dpg.show_item("training_ui::model_save_file_dialog"),
            )
            dpg.add_input_text(
                tag=MODEL_SAVE_PATH_INPUT_TAG,
                width=UI_INPUT_WIDTH_LONG - 70,
                default_value=global_train_config.MODEL_SAVE_PATH,
                hint="e.g., models/pilotnet_pytorch.pth",
            )

        dpg.add_spacer(height=5)
        dpg.add_text("Training Hyperparameters:")
        with dpg.group(horizontal=True):
            dpg.add_input_int(
                label="Batch Size",
                tag=BATCH_SIZE_INPUT_TAG,
                default_value=global_train_config.BATCH_SIZE,
                width=150,
            )
            dpg.add_input_int(
                label="Num Epochs",
                tag=NUM_EPOCHS_INPUT_TAG,
                default_value=global_train_config.NUM_EPOCHS,
                width=150,
            )
        with dpg.group(horizontal=True):
            dpg.add_input_float(
                label="Learning Rate",
                tag=LEARNING_RATE_INPUT_TAG,
                default_value=global_train_config.LEARNING_RATE,
                format="%.1e",
                width=150,
            )
            dpg.add_input_float(
                label="Validation Split",
                tag=VAL_SPLIT_SIZE_INPUT_TAG,
                default_value=global_train_config.VAL_SPLIT_SIZE,
                format="%.2f",
                min_value=0.0,
                max_value=0.99,
                step=0.01,
                width=150,
            )
        with dpg.group(horizontal=True):
            dpg.add_input_int(
                label="Random State",
                tag=RANDOM_STATE_INPUT_TAG,
                default_value=global_train_config.RANDOM_STATE,
                width=150,
            )
            dpg.add_input_int(
                label="Num Workers",
                tag=NUM_WORKERS_INPUT_TAG,
                default_value=global_train_config.NUM_WORKERS,
                min_value=0,
                width=150,
            )
        with dpg.group(horizontal=True):
            dpg.add_input_int(
                label="Early Stopping Patience",
                tag=EARLY_STOPPING_PATIENCE_INPUT_TAG,
                default_value=5,
                min_value=0,
                width=150,
            )
            dpg.add_input_float(
                label="Lambda Smoothness",
                tag=LAMBDA_SMOOTHNESS_INPUT_TAG,
                default_value=global_train_config.LAMBDA_SMOOTHNESS if hasattr(global_train_config, 'LAMBDA_SMOOTHNESS') else 0.01, # Use default from config if exists
                format="%.4f",
                min_value=0.0,
                step=0.001,
                width=150,
            )

        dpg.add_spacer(height=10)
        dpg.add_text("MLflow Logging Configuration:")
        dpg.add_checkbox(label="Enable MLflow Logging", tag=MLFLOW_ENABLE_CHECKBOX_TAG, default_value=True)
        dpg.add_input_text(label="MLflow Experiment Name", tag=MLFLOW_EXPERIMENT_NAME_INPUT_TAG,
                           default_value=global_train_config.MLFLOW_DEFAULT_EXPERIMENT_NAME, width=UI_INPUT_WIDTH_LONG -70) # Adjusted width
        dpg.add_input_text(label="MLflow Run Name (Optional)", tag=MLFLOW_RUN_NAME_INPUT_TAG,
                           hint="Leave blank for auto-generated", width=UI_INPUT_WIDTH_LONG - 70) # Adjusted width


        dpg.add_spacer(height=10)        
        dpg.add_button(
            label="Start Model Training",
            tag=TRAIN_BUTTON_TAG,
            callback=on_start_training_clicked,
            width=-1,
        )
        dpg.add_text("Status: Idle", tag=TRAINING_STATUS_TAG)

        dpg.add_spacer(height=5)
        dpg.add_text("Training Log:")
        dpg.add_input_text(
            tag=TRAINING_LOG_TAG,
            multiline=True,
            width=-1,
            height=300,
            readonly=True,
            default_value="Training logs will appear here...\n",
        )
