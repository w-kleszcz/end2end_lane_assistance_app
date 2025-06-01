import os
import cv2
import glob
import threading
import time
import queue
import numpy as np
import yaml
import dearpygui.dearpygui as dpg
from natsort import natsorted

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from model.data_loader import create_dataloaders_from_yaml
from model.model import build_pilotnet_model
import model.config as global_train_config # For default values


class Player:
    FRAME_DELAY_DEFAULT = 0.033  # ~30 FPS

    def __init__(self, texture_id, image_folder="", namespace=""):
        self.namespace = namespace

        if image_folder == "":
            self.image_paths = []
            self.images_folder = ""
        else:
            self.image_paths = natsorted(glob.glob(os.path.join(image_folder, "*.jpg")))
            self.images_folder = image_folder
            dpg.configure_item(
                self.tag_with_namespace("frame_slider"),
                max_value=len(self.get_valid_frame_indices()) - 1,
            )
            dpg.configure_item(self.tag_with_namespace("frame_slider"), enabled=True)

        self.texture_id = texture_id
        self.frame_delay = self.FRAME_DELAY_DEFAULT
        self.frame_index = 0
        self.playing = False

        self.images_idx_to_skip = []
        self.previous_slider_index = 0

        self.skip_start_idx = -1

    def tag_with_namespace(self, tag):
        if not self.namespace:
            return tag
        else:
            return self.namespace + "::" + tag

    def get_valid_frame_indices(self):
        return [
            i for i in range(len(self.image_paths)) if i not in self.images_idx_to_skip
        ]

    def get_number_of_valid_frames(self):
        return len(self.get_valid_frame_indices())

    def get_slider_index_from_frame_index(self, frame_index):
        valid_indices = self.get_valid_frame_indices()
        if frame_index in valid_indices:
            return valid_indices.index(frame_index)
        return 0  # fallback

    def get_frame_index_from_slider_index(self, slider_index):
        valid_indices = self.get_valid_frame_indices()
        if 0 <= slider_index < len(valid_indices):
            return valid_indices[slider_index]
        return 0  # fallback

    def load_frame(self, index):
        if index < 0 or index >= len(self.image_paths):
            return
        img = cv2.imread(self.image_paths[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        img = cv2.resize(img, (640, 480))
        img = img.astype(np.float32) / 255.0

        dpg.set_value(self.texture_id, img.flatten())

    def play_loop(self):
        while self.playing and self.get_number_of_valid_frames() > 0:
            self.load_frame(self.frame_index)
            print("Playing frame ", self.frame_index)

            valid_indices = self.get_valid_frame_indices()
            current_pos = self.get_slider_index_from_frame_index(self.frame_index)
            next_pos = (current_pos + 1) % len(valid_indices)
            self.frame_index = valid_indices[next_pos]

            self.set_slider_to_frame_index(self.frame_index)
            time.sleep(self.frame_delay)

    def on_folder_selected(self, sender, app_data, user_data):
        print("Selected folder: ", app_data["file_path_name"])
        self.image_paths = natsorted(
            glob.glob(os.path.join(app_data["file_path_name"], "*.jpg"))
        )
        self.images_folder = app_data["file_path_name"]
        self.images_idx_to_skip = []
        self.frame_index = 0
        dpg.configure_item(
            self.tag_with_namespace("frame_slider"),
            max_value=self.get_number_of_valid_frames() - 1,
        )
        dpg.set_value(self.tag_with_namespace("frame_slider"), 0)
        dpg.configure_item(self.tag_with_namespace("frame_slider"), enabled=True)
        dpg.set_value(
            self.tag_with_namespace("images_folder_path"), app_data["file_path_name"]
        )

    def on_player_slider_change(self, sender, app_data, user_data):
        self.on_pause()
        self.frame_index = self.get_frame_index_from_slider_index(app_data)
        self.load_frame(self.frame_index)
        self.previous_slider_index = app_data

    def on_play(self):
        if not self.playing and self.get_number_of_valid_frames() > 0:
            self.playing = True
            threading.Thread(target=self.play_loop, daemon=True).start()

    def on_pause(self):
        self.playing = False

    def on_step_forward(self):
        self.on_pause()
        valid_indices = self.get_valid_frame_indices()
        cur_idx = self.get_slider_index_from_frame_index(self.frame_index)
        if cur_idx + 1 < len(valid_indices):
            self.frame_index = valid_indices[cur_idx + 1]
        self.set_slider_to_frame_index(self.frame_index)
        self.load_frame(self.frame_index)

    def on_step_back(self):
        self.on_pause()
        valid_indices = self.get_valid_frame_indices()
        cur_idx = self.get_slider_index_from_frame_index(self.frame_index)
        if cur_idx - 1 >= 0:
            self.frame_index = valid_indices[cur_idx - 1]
        self.set_slider_to_frame_index(self.frame_index)
        self.load_frame(self.frame_index)

    def on_jump_n_frames_fwd(self):
        self.on_pause()
        valid_indices = self.get_valid_frame_indices()
        cur_idx = self.get_slider_index_from_frame_index(self.frame_index)
        if cur_idx + dpg.get_value(self.tag_with_namespace("n_frames_jump")) < len(
            valid_indices
        ):
            self.frame_index = valid_indices[
                cur_idx + dpg.get_value(self.tag_with_namespace("n_frames_jump"))
            ]
        self.set_slider_to_frame_index(self.frame_index)
        self.load_frame(self.frame_index)

    def on_jump_n_frames_bck(self):
        self.on_pause()
        valid_indices = self.get_valid_frame_indices()
        cur_idx = self.get_slider_index_from_frame_index(self.frame_index)
        if cur_idx - dpg.get_value(self.tag_with_namespace("n_frames_jump")) >= 0:
            self.frame_index = valid_indices[
                cur_idx - dpg.get_value(self.tag_with_namespace("n_frames_jump"))
            ]
        self.set_slider_to_frame_index(self.frame_index)
        self.load_frame(self.frame_index)

    def on_playback_speed_up(self):
        if self.frame_delay > 0.001:
            self.frame_delay *= 0.7

    def on_playback_speed_reset(self):
        self.frame_delay = self.FRAME_DELAY_DEFAULT

    def on_remove_frame(self):
        if (
            self.frame_index not in self.images_idx_to_skip
            and self.get_number_of_valid_frames() > 1
        ):
            current_valid_idx = self.get_slider_index_from_frame_index(self.frame_index)
            self.images_idx_to_skip.append(self.frame_index)

            valid_indices = self.get_valid_frame_indices()
            dpg.configure_item(
                self.tag_with_namespace("frame_slider"),
                max_value=len(valid_indices) - 1,
            )

            # Decide new frame index:
            if current_valid_idx < len(valid_indices):
                # Go to next valid frame if possible
                self.frame_index = valid_indices[
                    min(current_valid_idx, len(valid_indices) - 1)
                ]
            else:
                # If current_valid_idx is out of range (e.g. removed last), step back
                self.frame_index = valid_indices[-1]

            self.set_slider_to_frame_index(self.frame_index)
            self.load_frame(self.frame_index)

            print("Currently skipped frames: ", self.images_idx_to_skip)
            print(
                "Slider max ",
                dpg.get_item_configuration(self.tag_with_namespace("frame_slider"))[
                    "max_value"
                ],
            )

    def on_undo_last_frame_remove(self):
        if len(self.images_idx_to_skip) > 0:
            self.images_idx_to_skip.pop()
            dpg.configure_item(
                self.tag_with_namespace("frame_slider"),
                max_value=self.get_number_of_valid_frames() - 1,
            )
            self.set_slider_to_frame_index(self.frame_index)
            print("Currently skipped frames: ", self.images_idx_to_skip)

    def set_slider_to_frame_index(self, frame_index):
        slider_index = self.get_slider_index_from_frame_index(frame_index)
        dpg.set_value(self.tag_with_namespace("frame_slider"), slider_index)

    def increment_slider(self):
        slider_val = dpg.get_value(self.tag_with_namespace("frame_slider"))
        max_val = dpg.get_item_configuration(self.tag_with_namespace("frame_slider"))[
            "max_value"
        ]
        if slider_val < max_val:
            dpg.set_value(self.tag_with_namespace("frame_slider"), slider_val + 1)

    def decrement_slider(self):
        slider_val = dpg.get_value(self.tag_with_namespace("frame_slider"))
        if slider_val > 0:
            dpg.set_value(self.tag_with_namespace("frame_slider"), slider_val - 1)

    def on_skip_start(self):
        self.skip_start_idx = self.frame_index
        dpg.configure_item(self.tag_with_namespace("skip_frames_finish"), enabled=True)
        dpg.configure_item(self.tag_with_namespace("reset_skip_frames"), enabled=True)

    def on_reset_skip(self):
        self.skip_start_idx = -1
        dpg.configure_item(self.tag_with_namespace("skip_frames_finish"), enabled=False)
        dpg.configure_item(self.tag_with_namespace("reset_skip_frames"), enabled=False)

    def on_skip_finish(self):
        if self.skip_start_idx == -1:
            return

        start = min(self.skip_start_idx, self.frame_index)
        end = max(self.skip_start_idx, self.frame_index)

        # Range of frames to skip
        indices_to_skip = list(range(start, end + 1))
        print(f"Skipping frames from {start} to {end}")

        # Add to skip list (no duplicates)
        self.images_idx_to_skip.extend(
            [idx for idx in indices_to_skip if idx not in self.images_idx_to_skip]
        )

        # Reset skip range start
        self.skip_start_idx = -1

        # Recalculate valid frames
        valid_indices = self.get_valid_frame_indices()
        if not valid_indices:
            print("No valid frames remaining.")
            return

        # Find next valid frame after the old frame_index
        next_frame_candidates = [i for i in valid_indices if i > end]
        if next_frame_candidates:
            self.frame_index = next_frame_candidates[0]
        else:
            # If no frames after end, go to last available valid frame
            self.frame_index = valid_indices[-1]

        # Update slider and display
        dpg.configure_item(
            self.tag_with_namespace("frame_slider"), max_value=len(valid_indices) - 1
        )
        self.set_slider_to_frame_index(self.frame_index)
        self.load_frame(self.frame_index)

        # Disable skip controls
        dpg.configure_item(self.tag_with_namespace("skip_frames_finish"), enabled=False)
        dpg.configure_item(self.tag_with_namespace("reset_skip_frames"), enabled=False)

        print("Currently skipped frames: ", self.images_idx_to_skip)
        print(
            "Slider max ",
            dpg.get_item_configuration(self.tag_with_namespace("frame_slider"))[
                "max_value"
            ],
        )


class DatasetPreparator(Player):
    def __init__(
        self, texture_id, image_folder="", image_annotations_file="", namespace=""
    ):
        super().__init__(texture_id, image_folder, namespace)
        self.image_annotations_file = image_annotations_file

        self.test_set_idx_start = None
        self.test_set_idx_finish = None

        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            callback=self.save_file_with_dataset_metadata,
            id=self.tag_with_namespace("save_file_dialog"),
            width=500,
            height=400,
        ):
            dpg.add_file_extension(".yaml")

        with dpg.window(
            label="Error",
            modal=True,
            show=False,
            tag=self.tag_with_namespace("error_popup"),
            no_title_bar=True,
            width=400,
            height=100,
        ):
            dpg.add_text(
                "", tag=self.tag_with_namespace("error_message_text")
            )  # Placeholder for the message
            dpg.add_spacer(height=10)
            dpg.add_button(
                label="Close",
                width=75,
                callback=lambda: dpg.hide_item(self.tag_with_namespace("error_popup")),
            )

    def set_raw_data_folder(self, raw_data_folder):
        self.raw_data_folder = raw_data_folder

    def on_image_annotations_file_selected(self, sender, app_data, user_data):
        print("Selected image annotations file: ", app_data["file_path_name"])
        self.image_annotations_file = app_data["file_path_name"]
        dpg.set_value("preparator::annotations_file_path", app_data["file_path_name"])

    def on_test_dataset_start(self):
        self.test_set_idx_start = self.frame_index
        print("Test set start index set to ", self.test_set_idx_start)

    def on_test_dataset_finish(self):
        self.test_set_idx_finish = self.frame_index
        print("Test set finish index set to ", self.test_set_idx_finish)

    def on_save_dataset(self, path):
        if self.image_annotations_file == "":
            self.show_error_popup("Please select image annotations file.")
            return

        if self.images_folder == "":
            self.show_error_popup("Please select raw images folder.")
            return

        if self.test_set_idx_start is None or self.test_set_idx_finish is None:
            self.show_error_popup(
                "Please set start and finish indices for the test set."
            )
            return

        dpg.show_item(self.tag_with_namespace("save_file_dialog"))

    def save_file_with_dataset_metadata(self, sender, app_data):
        data = {
            "annotations_file": self.image_annotations_file,
            "images_dir": self.images_folder,
            "indices_to_skip": self.images_idx_to_skip,
            "test_set_idx_start": self.test_set_idx_start,
            "test_set_idx_end": self.test_set_idx_finish,
        }
        with open(app_data["file_path_name"], "w") as f:
            yaml.dump(data, f, default_flow_style=False)

    def show_error_popup(self, message):
        dpg.set_value(self.tag_with_namespace("error_message_text"), message)
        dpg.set_item_pos(self.tag_with_namespace("error_popup"), (150, 150))
        dpg.show_item(self.tag_with_namespace("error_popup"))


APP_WINDOW_WIDTH = 660
APP_WINDOW_HEIGHT = 800
UI_INPUT_WIDTH_LONG = 550

# --- Globals for Model Training ---
training_log_queue = queue.Queue()
training_log_tag = "training_log_text_area"
training_status_tag = "training_status_indicator"
train_button_tag = "train_model_button_tag"
data_config_path_input_tag = "data_config_path_input_tag"
plot_save_path_input_tag = "plot_save_path_input_tag"
model_save_path_input_tag = "model_save_path_input_tag"
batch_size_input_tag = "batch_size_input_tag"
num_epochs_input_tag = "num_epochs_input_tag"
learning_rate_input_tag = "learning_rate_input_tag"
val_split_size_input_tag = "val_split_size_input_tag"
random_state_input_tag = "random_state_input_tag"
num_workers_input_tag = "num_workers_input_tag"
early_stopping_patience_input_tag = "early_stopping_patience_input_tag"


# ---------- Create texture buffer ----------
dpg.create_context()

dpg.create_viewport(
    title="End2end Lane Assistance App",
    width=APP_WINDOW_WIDTH,
    height=APP_WINDOW_HEIGHT,
)

with dpg.texture_registry():
    dummy_image = np.zeros((480, 640, 4), dtype=np.float32)

    preparator_player_texture_id = dpg.generate_uuid()
    dpg.add_dynamic_texture(
        640, 480, dummy_image.flatten(), tag=preparator_player_texture_id
    )

    evaluator_player_texture_id = dpg.generate_uuid()
    dpg.add_dynamic_texture(
        640, 480, dummy_image.flatten(), tag=evaluator_player_texture_id
    )


# ---------- Model Training Functions ----------
def log_to_gui(message):
    """Puts a message into the training log queue."""
    training_log_queue.put(message)


def format_log_message(epoch, logs, total_epochs):
    """Formats the log message for a Keras epoch."""
    loss_val = logs.get("loss")
    val_loss = logs.get("val_loss")

    # Format loss, handling potential None
    loss_str = f"{loss_val:.4f}" if loss_val is not None else "N/A"

    # Format val_loss, handling potential None
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

        # --- 1. Data Preparation ---
        log_to_gui("Loading and preparing data...\n")
        # Note: create_dataloaders_from_yaml might print, consider capturing or modifying it
        train_loader, val_loader, test_loader = create_dataloaders_from_yaml(
            data_config_path,
            batch_size=batch_size,
            val_split_size=val_split_size,
            random_state=random_state,
            num_workers=num_workers,
        )

        if not train_loader and not val_loader and not test_loader:
            log_to_gui("ERROR: Failed to create any DataLoaders. Aborting.\n")
            dpg.set_value(training_status_tag, "Error: Data loading failed")
            return
        if not train_loader:
            log_to_gui(
                "ERROR: Failed to create training DataLoader. Aborting training.\n"
            )
            dpg.set_value(training_status_tag, "Error: Training data missing")
            return
        if len(train_loader) == 0:
            log_to_gui(
                "ERROR: train_loader is empty. Check annotations, image paths, and BATCH_SIZE.\n"
            )
            dpg.set_value(training_status_tag, "Error: Training data empty")
            return
        log_to_gui("Data loaded successfully.\n")

        # --- 2. Model Building ---
        log_to_gui("Building the PilotNet model...\n")
        # Use IMG_CHANNELS for PyTorch model constructor
        pilotnet_model = build_pilotnet_model(input_channels=global_train_config.IMG_CHANNELS)
        pilotnet_model.to(device)
        log_to_gui("Model built.\n")
        log_to_gui("Model Summary:\n" + str(pilotnet_model) + "\n")

        # --- 3. Model Compilation ---
        log_to_gui("Setting up optimizer and loss function...\n")
        optimizer = optim.Adam(pilotnet_model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        log_to_gui("Optimizer and loss function set up.\n")
        
        # --- 4. Model Training ---
        log_to_gui(f"Starting training for {num_epochs} epochs...\n")
        dpg.set_value(training_status_tag, "Status: Training...")

        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

        best_val_loss = float('inf')
        epochs_no_improve = 0
        stopped_epoch_num = 0 # For logging early stopping

        for epoch in range(num_epochs):
            pilotnet_model.train() # Set model to training mode
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
            current_logs = {'loss': avg_train_loss}

            # Validation loop
            avg_val_loss = None
            if val_loader and len(val_loader) > 0:
                pilotnet_model.eval() # Set model to evaluation mode
                val_running_loss = 0.0
                with torch.no_grad():
                    for inputs_val, labels_val in val_loader:
                        inputs_val, labels_val = inputs_val.to(device), labels_val.to(device)
                        outputs_val = pilotnet_model(inputs_val)
                        loss_val = criterion(outputs_val, labels_val)
                        val_running_loss += loss_val.item()
                avg_val_loss = val_running_loss / len(val_loader)
                current_logs['val_loss'] = avg_val_loss
                log_to_gui(format_log_message(epoch, current_logs, num_epochs))

                # ModelCheckpoint (save_best_only) logic
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(pilotnet_model.state_dict(), model_save_path)
                    log_to_gui(f"Epoch {epoch+1}: val_loss improved to {avg_val_loss:.4f}, model saved to {model_save_path}\n")
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    log_to_gui(f"Epoch {epoch+1}: val_loss ({avg_val_loss:.4f}) did not improve from {best_val_loss:.4f}.\n")
                
                # EarlyStopping logic
                if early_stopping_patience > 0 and epochs_no_improve >= early_stopping_patience:
                    stopped_epoch_num = epoch + 1
                    log_to_gui(f"Early stopping triggered after {epochs_no_improve} epochs without improvement.\n")
                    # Best model is already saved by the logic above.
                    # If we need to ensure the `pilotnet_model` variable holds these best weights:
                    # pilotnet_model.load_state_dict(torch.load(model_save_path, map_location=device))
                    # log_to_gui("Restored best model weights to current model instance.\n")
                    break 
            else: # No validation loader
                log_to_gui(format_log_message(epoch, current_logs, num_epochs))
                # If no validation, save model at the end of all epochs
                if epoch == num_epochs - 1:
                     torch.save(pilotnet_model.state_dict(), model_save_path)
                     log_to_gui(f"Model saved at end of training (no validation) to {model_save_path}\n")

        log_to_gui("Training completed.\n")

        # --- 5. Model Saving ---
        # Logic for saving the best model is now integrated into the training loop.
        if stopped_epoch_num > 0:
            log_to_gui(f"Early stopping was triggered at epoch {stopped_epoch_num}.\n")
        
        if os.path.exists(model_save_path):
            log_to_gui(f"Best model (based on val_loss) during training was saved to: {model_save_path}\n")
        else:
            log_to_gui(f"No model was saved (e.g. no validation or training did not complete an epoch to save).\n")

        # --- 6. Model Evaluation on Test Dataset ---
        if test_loader and len(test_loader) > 0:
            log_to_gui("Evaluating model on the test set...\n")
            preds = []
            truths = []
            for images, labels in test_loader:
                # Load the best model for evaluation
                eval_model = build_pilotnet_model(input_channels=global_train_config.IMG_CHANNELS)
                if os.path.exists(model_save_path):
                    log_to_gui(f"Loading best model from {model_save_path} for evaluation.\n")
                    eval_model.load_state_dict(torch.load(model_save_path, map_location=device))
                else:
                    log_to_gui("Warning: Saved model not found for evaluation. Using model from end of training (if available).\n")
                    eval_model.load_state_dict(pilotnet_model.state_dict()) # Use the one from training loop
                
                eval_model.to(device)
                eval_model.eval()
                with torch.no_grad():
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
                log_to_gui(f"Created directory for plot: {plot_dir}\n")
            plt.savefig(plot_save_path)
            log_to_gui(f"Evaluation plot saved to {plot_save_path}\n")
            plt.close()
        else:
            log_to_gui(
                "INFO: No test_loader available to perform evaluation after training.\n"
            )

        log_to_gui("--- Training Process Finished Successfully ---\n")
        dpg.set_value(training_status_tag, "Status: Completed")

    except Exception as e:
        log_to_gui(f"\n--- ERROR during training process ---\n{str(e)}\n")
        import traceback

        log_to_gui(traceback.format_exc() + "\n")
        dpg.set_value(training_status_tag, "Status: Error")
    finally:
        dpg.configure_item(train_button_tag, enabled=True)


def on_start_training_clicked():
    data_config = dpg.get_value(data_config_path_input_tag)
    plot_path = dpg.get_value(plot_save_path_input_tag)
    model_save_path = dpg.get_value(model_save_path_input_tag)
    batch_size = dpg.get_value(batch_size_input_tag)
    num_epochs = dpg.get_value(num_epochs_input_tag)
    learning_rate = dpg.get_value(learning_rate_input_tag)
    val_split_size = dpg.get_value(val_split_size_input_tag)
    random_state = dpg.get_value(random_state_input_tag)
    num_workers = dpg.get_value(num_workers_input_tag)
    early_stopping_patience = dpg.get_value(early_stopping_patience_input_tag)

    if not data_config or not os.path.isfile(data_config):
        log_to_gui("ERROR: Data Config YAML path is invalid or file does not exist.\n")
        dpg.set_value(training_status_tag, "Error: Invalid Data Config Path")
        return
    if not plot_path:
        log_to_gui("ERROR: Plot Save Path is required.\n")
        dpg.set_value(training_status_tag, "Error: Invalid Plot Save Path")
        return
    if not model_save_path:
        log_to_gui("ERROR: Model Save Path is required.\n")
        dpg.set_value(training_status_tag, "Error: Invalid Model Save Path")
        return
    # Basic validation for numeric types (can be expanded)
    if not isinstance(batch_size, int) or batch_size <= 0:
        log_to_gui(f"ERROR: Invalid Batch Size ({batch_size}). Must be a positive integer.\n")
        return
    if not isinstance(num_epochs, int) or num_epochs <= 0:
        log_to_gui(f"ERROR: Invalid Number of Epochs ({num_epochs}). Must be a positive integer.\n")
        return
    if not isinstance(learning_rate, float) or learning_rate <= 0:
        log_to_gui(f"ERROR: Invalid Learning Rate ({learning_rate}). Must be a positive float.\n")
        return
    if not isinstance(val_split_size, float) or not (0.0 <= val_split_size < 1.0):
        log_to_gui(f"ERROR: Invalid Validation Split Size ({val_split_size}). Must be a float between 0.0 and 1.0 (exclusive of 1.0).\n")
        return
    if not isinstance(early_stopping_patience, int) or early_stopping_patience < 0: # 0 means no early stopping if patience is 0
        log_to_gui(f"ERROR: Invalid Early Stopping Patience ({early_stopping_patience}). Must be a non-negative integer.\n")
        return

    dpg.set_value(training_log_tag, "")  # Clear previous logs
    dpg.set_value(training_status_tag, "Status: Starting...")
    dpg.configure_item(train_button_tag, enabled=False)

    thread = threading.Thread(
        target=run_training_thread,
        args=(data_config, plot_path, model_save_path,
              batch_size, num_epochs, learning_rate,
              val_split_size, random_state, num_workers,
              early_stopping_patience),
        daemon=True,
    )
    thread.start()


def update_training_log_display():
    """Checks queue and updates the log display."""
    try:
        while not training_log_queue.empty():
            message = training_log_queue.get_nowait()
            current_log = dpg.get_value(training_log_tag)
            dpg.set_value(training_log_tag, current_log + message)
            # TODO: Add auto-scrolling
    except queue.Empty:
        pass


def select_data_config_callback(sender, app_data):
    dpg.set_value(data_config_path_input_tag, app_data["file_path_name"])


def select_plot_save_path_callback(sender, app_data):
    dpg.set_value(plot_save_path_input_tag, app_data["file_path_name"])

def select_model_save_path_callback(sender, app_data):
    dpg.set_value(model_save_path_input_tag, app_data["file_path_name"])


# ---------- GUI Layout ----------
with dpg.window(
    label="End2end Lane Assistance App",
    width=APP_WINDOW_WIDTH,
    height=APP_WINDOW_HEIGHT,
):
    with dpg.tab_bar():

        with dpg.tab(label="Data Preparation"):

            dataset_preparator = DatasetPreparator(
                preparator_player_texture_id, namespace="preparator"
            )

            # File and Folder Browsing Controls
            dpg.add_text("Raw images folder:")
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Browse",
                    callback=lambda: dpg.show_item("preparator::folder_dialog"),
                )  # Button for browsing folder
                dpg.add_input_text(
                    tag="preparator::images_folder_path",
                    width=UI_INPUT_WIDTH_LONG,
                    readonly=True,
                )  # Textbox for folder path

            dpg.add_text("Raw annotations file:")
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Browse",
                    callback=lambda: dpg.show_item("preparator::file_dialog"),
                )
                dpg.add_input_text(
                    tag="preparator::annotations_file_path",
                    width=UI_INPUT_WIDTH_LONG,
                    readonly=True,
                )  # Textbox for file path

            # Folder Dialog (for selecting folders)
            with dpg.file_dialog(
                directory_selector=True,
                show=False,
                tag="preparator::folder_dialog",
                callback=dataset_preparator.on_folder_selected,
                width=500,
                height=400,
            ):
                dpg.add_file_extension(".*")

            # File Dialog (for selecting a single file)
            with dpg.file_dialog(
                directory_selector=False,
                show=False,
                tag="preparator::file_dialog",
                callback=dataset_preparator.on_image_annotations_file_selected,
                width=500,
                height=400,
            ):
                dpg.add_file_extension(".txt")

            dpg.add_image(preparator_player_texture_id)

            dpg.add_slider_int(
                tag="preparator::frame_slider",
                label="",
                min_value=0,
                max_value=0,
                default_value=0,
                width=640,
                callback=dataset_preparator.on_player_slider_change,
                enabled=True,
            )

            with dpg.group(horizontal=True):
                dpg.add_button(label="Play", callback=dataset_preparator.on_play)
                dpg.add_button(label="Pause", callback=dataset_preparator.on_pause)
                dpg.add_button(
                    label="Step Back", callback=dataset_preparator.on_step_back
                )
                dpg.add_button(
                    label="Step Forward", callback=dataset_preparator.on_step_forward
                )
                dpg.add_button(
                    label="Speed Up", callback=dataset_preparator.on_playback_speed_up
                )
                dpg.add_button(
                    label="Speed Reset",
                    callback=dataset_preparator.on_playback_speed_reset,
                )

            with dpg.group(horizontal=True):
                dpg.add_text("N =")
                dpg.add_input_int(
                    tag="preparator::n_frames_jump", width=100, default_value=100
                )
                dpg.add_button(
                    label="Jump N Frames Back",
                    tag="preparator::jump_n_frames_bck",
                    callback=dataset_preparator.on_jump_n_frames_bck,
                )
                dpg.add_button(
                    label="Jump N Frames Forward",
                    tag="preparator::jump_n_frames_fwd",
                    callback=dataset_preparator.on_jump_n_frames_fwd,
                )

            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Remove from this frame...",
                    tag="preparator::skip_frames_start",
                    callback=dataset_preparator.on_skip_start,
                )
                dpg.add_button(
                    label="Remove to this frame",
                    tag="preparator::skip_frames_finish",
                    callback=dataset_preparator.on_skip_finish,
                    enabled=False,
                )

                dpg.add_button(
                    label="Reset range removal",
                    tag="preparator::reset_skip_frames",
                    callback=dataset_preparator.on_reset_skip,
                    enabled=False,
                )

            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Remove frame", callback=dataset_preparator.on_remove_frame
                )
                dpg.add_button(
                    label="Undo last remove",
                    callback=dataset_preparator.on_undo_last_frame_remove,
                )
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Set frame for test dataset start",
                    callback=dataset_preparator.on_test_dataset_start,
                )
                dpg.add_button(
                    label="Set frame for test dataset finish",
                    callback=dataset_preparator.on_test_dataset_finish,
                )

            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Save dataset", callback=dataset_preparator.on_save_dataset
                )

        with dpg.tab(label="Model Training"):
            dpg.add_text(
                "Configure and start model training based on a data YAML file."
            )

            # File Dialog for Data Config YAML
            with dpg.file_dialog(
                directory_selector=False,
                show=False,
                callback=select_data_config_callback,
                tag="data_config_file_dialog",
                width=500,
                height=400,
            ):
                dpg.add_file_extension(".yaml", color=(255, 255, 0, 255))
                dpg.add_file_extension(".yml", color=(255, 255, 0, 255))

            dpg.add_text("Data Configuration YAML:")
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Browse##DataConfig",
                    callback=lambda: dpg.show_item("data_config_file_dialog"),
                )
                dpg.add_input_text(
                    tag=data_config_path_input_tag,
                    width=UI_INPUT_WIDTH_LONG - 70,
                    hint="e.g., new_data/data.yaml",
                )

            with dpg.file_dialog(
                directory_selector=False,
                show=False,
                callback=select_plot_save_path_callback,
                tag="plot_save_file_dialog",
                width=500,
                height=400,
                default_filename="steering_angle_evaluation.png",
            ):
                dpg.add_file_extension(".png", color=(0, 255, 0, 255))
                dpg.add_file_extension(".*")

            dpg.add_text("Evaluation Plot Save Path:")
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Browse##PlotSave",
                    callback=lambda: dpg.show_item("plot_save_file_dialog"),
                )
                dpg.add_input_text(
                    tag=plot_save_path_input_tag,
                    width=UI_INPUT_WIDTH_LONG - 70,
                    default_value="plots/steering_angle_evaluation.png",
                    hint="e.g., plots/evaluation.png",
                )

            # File Dialog for Model Save Path
            with dpg.file_dialog(
                directory_selector=False,
                show=False,
                callback=select_model_save_path_callback,
                tag="model_save_file_dialog",
                width=500,
                height=400,
                default_filename="pilotnet_model.keras",
            ):
                dpg.add_file_extension(".keras", color=(0, 255, 255, 255))
                dpg.add_file_extension(".*")

            dpg.add_text("Model Save Path:")
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Browse##ModelSave",
                    callback=lambda: dpg.show_item("model_save_file_dialog"),
                )
                dpg.add_input_text(
                    tag=model_save_path_input_tag,
                    width=UI_INPUT_WIDTH_LONG - 70,
                    default_value=global_train_config.MODEL_SAVE_PATH,
                    hint="e.g., models/pilotnet.keras",
                )

            dpg.add_spacer(height=5)
            dpg.add_text("Training Hyperparameters:")
            with dpg.group(horizontal=True):
                dpg.add_input_int(
                    label="Batch Size",
                    tag=batch_size_input_tag,
                    default_value=global_train_config.BATCH_SIZE,
                    width=150,
                )
                dpg.add_input_int(
                    label="Num Epochs",
                    tag=num_epochs_input_tag,
                    default_value=global_train_config.NUM_EPOCHS,
                    width=150,
                )
            with dpg.group(horizontal=True):
                dpg.add_input_float(
                    label="Learning Rate",
                    tag=learning_rate_input_tag,
                    default_value=global_train_config.LEARNING_RATE,
                    format="%.1e", # Scientific notation
                    width=150,
                )
                dpg.add_input_float(
                    label="Validation Split",
                    tag=val_split_size_input_tag,
                    default_value=global_train_config.VAL_SPLIT_SIZE,
                    format="%.2f",
                    min_value=0.0, max_value=0.99, step=0.01, # Max < 1.0
                    width=150,
                )
            with dpg.group(horizontal=True):
                dpg.add_input_int(
                    label="Random State",
                    tag=random_state_input_tag,
                    default_value=global_train_config.RANDOM_STATE,
                    width=150,
                )
                dpg.add_input_int(
                    label="Num Workers",
                    tag=num_workers_input_tag,
                    default_value=global_train_config.NUM_WORKERS,
                    min_value=0,
                    width=150,
                )
            with dpg.group(horizontal=True):
                dpg.add_input_int(
                    label="Early Stopping Patience",
                    tag=early_stopping_patience_input_tag,
                    default_value=5, # Default patience
                    min_value=0, # 0 could mean no early stopping depending on Keras interpretation or disable it
                    width=150,
                )

            dpg.add_spacer(height=10)
            dpg.add_button(
                label="Start Model Training",
                tag=train_button_tag,
                callback=on_start_training_clicked,
                width=-1,
            )
            dpg.add_text("Status: Idle", tag=training_status_tag)

            dpg.add_spacer(height=5)
            dpg.add_text("Training Log:")
            dpg.add_input_text(
                tag=training_log_tag,
                multiline=True,
                width=-1,
                height=300,
                readonly=True,
                default_value="Training logs will appear here...\n",
            )

            # Timer to update logs from queue - this is one way, another is frame callback
            # dpg.add_timer_callback(update_training_log_display, delay=100, parent=dpg.last_item()) # if tab is item

        with dpg.tab(label="Model Evaluation"):
            dpg.add_text("Model evaluation will be implemented here.")
            dpg.add_button(
                label="Evaluate Model",
                callback=lambda: print("Model evaluation started."),
            )

        with dpg.tab(label="Model Deployment"):
            dpg.add_text("Model deployment will be implemented here.")
            dpg.add_button(
                label="Deploy Model",
                callback=lambda: print("Model deployment started."),
            )

# ---------- Launch ----------
dpg.setup_dearpygui()
dpg.show_viewport()

# --- Explicit Render Loop (Replaces start_dearpygui) ---
while dpg.is_dearpygui_running():
    # Call your function directly within the loop.
    # It will run once every frame.
    update_training_log_display()

    # This is crucial: Renders the frame
    dpg.render_dearpygui_frame()

# This runs only after the window is closed
dpg.destroy_context()
