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

    def __init__(self, image_folder, texture_id):
        self.image_paths = natsorted(glob.glob(os.path.join(image_folder, "*.jpg")))
        self.texture_id = texture_id
        self.frame_delay = 0.033
        self.frame_index = 0
        self.playing = False

    def load_frame(self, index):
        if index < 0 or index >= len(self.image_paths):
            return
        img = cv2.imread(self.image_paths[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        img = cv2.resize(img, (640, 480))
        img = img.astype(np.float32) / 255.0

        dpg.set_value(self.texture_id, img.flatten())

    def play_loop(self):
        while self.playing:
            self.load_frame(self.frame_index)
            self.frame_index = (self.frame_index + 1) % len(self.image_paths)
            time.sleep(self.frame_delay)

    def on_play(self):
        if not self.playing:
            self.playing = True
            threading.Thread(target=self.play_loop, daemon=True).start()

    def on_pause(self):
        self.playing = False

    def on_step_forward(self):
        self.on_pause()
        self.frame_index = (self.frame_index + 1) % len(self.image_paths)
        self.load_frame(self.frame_index)

    def on_step_back(self):
        self.on_pause()
        self.frame_index = (self.frame_index - 1) % len(self.image_paths)
        self.load_frame(self.frame_index)

    def on_playback_speed_up(self):
        if self.frame_delay <= 0.001:
            return

        self.frame_delay *= 0.7

    def on_playback_speed_reset(self):
        self.frame_delay = self.FRAME_DELAY_DEFAULT


class DatasetPreparator(Player):
    def __init__(self, image_folder, image_annotations_file, texture_id):
        super().__init__(image_folder, texture_id)
        self.image_annotations_file = image_annotations_file
        self.images_to_remove = []
        self.test_set_start = None
        self.test_set_finish = None

    def on_remove_frame():
        pass

    def on_undo_last_frame_remove():
        pass

    def on_test_dataset_start():
        pass

    def on_test_dataset_finish():
        pass

    def on_save_dataset(path):
        pass


def folder_selected_callback(sender, app_data):
    # Update folder path input field with the selected folder path
    dpg.set_value("folder_path", app_data["file_path_name"])


def file_selected_callback(sender, app_data):
    # Update file path input field with the selected file path
    dpg.set_value("file_path", app_data["file_path_name"])


# ---------- Create texture buffer ----------
dpg.create_context()

dpg.create_viewport(title="End2end Lane Assistance App", width=800, height=600)

with dpg.texture_registry():
    texture_id = dpg.generate_uuid()
    dummy_image = np.zeros((480, 640, 4), dtype=np.float32)
    dpg.add_dynamic_texture(640, 480, dummy_image.flatten(), tag=texture_id)


player = Player("data/raw/07012018/data", texture_id)


# ---------- GUI Layout ----------
with dpg.window(label="End2end Lane Assistance App", width=700, height=550):
    with dpg.tab_bar():

        with dpg.tab(label="Data Preparation"):

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
                callback=folder_selected_callback,
            ):
                dpg.add_file_extension(".*")  # Allow any file type

            # File Dialog (for selecting a single file)
            with dpg.file_dialog(
                directory_selector=False,
                show=False,
                tag="file_dialog",
                callback=file_selected_callback,
            ):
                dpg.add_file_extension(".*")  # Allow any file type

            dpg.add_image(texture_id)

            with dpg.group(horizontal=True):
                dpg.add_button(label="Play", callback=player.on_play)
                dpg.add_button(label="Pause", callback=player.on_pause)
                dpg.add_button(label="Step Forward", callback=player.on_step_forward)
                dpg.add_button(label="Step Back", callback=player.on_step_back)
                dpg.add_button(label="Speed Up", callback=player.on_playback_speed_up)
                dpg.add_button(
                    label="Speed Reset", callback=player.on_playback_speed_reset
                )

            with dpg.group(horizontal=True):
                dpg.add_button(label="Remove frame", callback=None)
                dpg.add_button(label="Undo last remove", callback=None)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Set frame for test dataset start", callback=None)
                dpg.add_button(label="Set frame for test dataset finish", callback=None)

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
player.load_frame(0)  # Load the first frame initially
dpg.start_dearpygui()
dpg.destroy_context()
