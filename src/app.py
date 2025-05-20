import os
import cv2
import glob
import threading
import time
import numpy as np
import dearpygui.dearpygui as dpg
from natsort import natsorted


class Player:
    FRAME_DELAY_DEFAULT = 0.033  # ~30 FPS

    def __init__(self, texture_id, image_folder=""):
        if image_folder == "":
            self.image_paths = []
        else:
            self.image_paths = natsorted(glob.glob(os.path.join(image_folder, "*.jpg")))
            dpg.configure_item(
                "frame_slider", max_value=len(self.get_valid_frame_indices()) - 1
            )
            dpg.configure_item("frame_slider", enabled=True)

        self.texture_id = texture_id
        self.frame_delay = self.FRAME_DELAY_DEFAULT
        self.frame_index = 0
        self.playing = False

        self.images_idx_to_skip = []
        self.previous_slider_index = 0

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
        self.images_idx_to_skip = []
        self.frame_index = 0
        dpg.configure_item(
            "frame_slider", max_value=self.get_number_of_valid_frames() - 1
        )
        dpg.set_value("frame_slider", 0)
        dpg.configure_item("frame_slider", enabled=True)

    def on_player_slider_change(self, sender, app_data, user_data):
        self.on_pause()
        print("Raw slider value is ", app_data)
        self.frame_index = self.get_frame_index_from_slider_index(app_data)
        self.load_frame(self.frame_index)
        self.previous_slider_index = app_data
        print("Set frame index via slider to ", self.frame_index)

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
            dpg.configure_item("frame_slider", max_value=len(valid_indices) - 1)

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
                "Slider max ", dpg.get_item_configuration("frame_slider")["max_value"]
            )

    def on_undo_last_frame_remove(self):
        if len(self.images_idx_to_skip) > 0:
            self.images_idx_to_skip.pop()
            dpg.configure_item(
                "frame_slider", max_value=self.get_number_of_valid_frames() - 1
            )
            self.set_slider_to_frame_index(self.frame_index)
            print("Currently skipped frames: ", self.images_idx_to_skip)

    def set_slider_to_frame_index(self, frame_index):
        slider_index = self.get_slider_index_from_frame_index(frame_index)
        dpg.set_value("frame_slider", slider_index)

    def increment_slider(self):
        slider_val = dpg.get_value("frame_slider")
        max_val = dpg.get_item_configuration("frame_slider")["max_value"]
        if slider_val < max_val:
            dpg.set_value("frame_slider", slider_val + 1)

    def decrement_slider(self):
        slider_val = dpg.get_value("frame_slider")
        if slider_val > 0:
            dpg.set_value("frame_slider", slider_val - 1)


class DatasetPreparator(Player):
    def __init__(self, texture_id, image_folder="", image_annotations_file=""):
        super().__init__(texture_id, image_folder)
        self.image_annotations_file = image_annotations_file

        self.test_set_idx_start = None
        self.test_set_idx_finish = None

        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            modal=True,
            tag="save_file_dialog",
            callback=self.save_file_with_dataset_metadata,
            width=400,
            height=300,
            default_filename="output.txt",
            file_count=1,
            file_dialog_mode=dpg.mvFileDialogMode.Save,
        ):
            dpg.add_file_extension(".*")

    def set_raw_data_folder(self, raw_data_folder):
        self.raw_data_folder = raw_data_folder

    def on_image_annotations_file_selected(self, sender, app_data, user_data):
        print("Selected image annotations file: ", app_data["file_path_name"])
        self.image_annotations_file = app_data["file_path_name"]

    def on_test_dataset_start(self):
        self.test_set_idx_start = self.frame_index
        print("Test set start index set to ", self.test_set_idx_start)

    def on_test_dataset_finish(self):
        self.test_set_idx_finish = self.frame_index
        print("Test set finish index set to ", self.test_set_idx_finish)

    def on_save_dataset(path):
        pass

    def save_file_with_dataset_metadata():
        dpg.show_item("save_file_dialog")
        pass


# ---------- Create texture buffer ----------
dpg.create_context()

dpg.create_viewport(title="End2end Lane Assistance App", width=800, height=600)

with dpg.texture_registry():
    player_texture_id = dpg.generate_uuid()
    dummy_image = np.zeros((480, 640, 4), dtype=np.float32)
    dpg.add_dynamic_texture(640, 480, dummy_image.flatten(), tag=player_texture_id)


# ---------- GUI Layout ----------
with dpg.window(label="End2end Lane Assistance App", width=700, height=550):
    with dpg.tab_bar():

        with dpg.tab(label="Data Preparation"):

            dataset_preparator = DatasetPreparator(player_texture_id)

            # File and Folder Browsing Controls
            dpg.add_text("Raw images folder:")
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Browse", callback=lambda: dpg.show_item("folder_dialog")
                )  # Button for browsing folder
                dpg.add_input_text(
                    tag="folder_path", width=500, readonly=True
                )  # Textbox for folder path

            dpg.add_text("Raw annotations file:")
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Browse", callback=lambda: dpg.show_item("file_dialog")
                )
                dpg.add_input_text(
                    tag="file_path", width=500, readonly=True
                )  # Textbox for file path

            dpg.add_button(label="Set raw dataset", callback=None)

            # Folder Dialog (for selecting folders)
            with dpg.file_dialog(
                directory_selector=True,
                show=False,
                tag="folder_dialog",
                callback=dataset_preparator.on_folder_selected,
            ):
                dpg.add_file_extension(".*")  # Allow any file type

            # File Dialog (for selecting a single file)
            with dpg.file_dialog(
                directory_selector=False,
                show=False,
                tag="file_dialog",
                callback=dataset_preparator.on_image_annotations_file_selected,
            ):
                dpg.add_file_extension(".*")  # Allow any file type

            dpg.add_image(player_texture_id)

            dpg.add_slider_int(
                tag="frame_slider",
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
                    label="Step Forward", callback=dataset_preparator.on_step_forward
                )
                dpg.add_button(
                    label="Step Back", callback=dataset_preparator.on_step_back
                )
                dpg.add_button(
                    label="Speed Up", callback=dataset_preparator.on_playback_speed_up
                )
                dpg.add_button(
                    label="Speed Reset",
                    callback=dataset_preparator.on_playback_speed_reset,
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
            dpg.add_text("Model training will be implemented here.")
            dpg.add_button(
                label="Train Model", callback=lambda: print("Model training started.")
            )

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

dpg.start_dearpygui()
dpg.destroy_context()
