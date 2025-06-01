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
        if not self.image_paths or index < 0 or index >= len(self.image_paths):
            return
        img = cv2.imread(self.image_paths[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        img = cv2.resize(img, (640, 480))
        img = img.astype(np.float32) / 255.0

        dpg.set_value(self.texture_id, img.flatten())

    def play_loop(self):
        while self.playing and self.get_number_of_valid_frames() > 0:
            self.load_frame(self.frame_index)
            # print("Playing frame ", self.frame_index) # Optional: for debugging

            valid_indices = self.get_valid_frame_indices()
            if not valid_indices: # Should not happen if get_number_of_valid_frames > 0
                self.playing = False
                break
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
            max_value=self.get_number_of_valid_frames() - 1 if self.get_number_of_valid_frames() > 0 else 0,
        )
        dpg.set_value(self.tag_with_namespace("frame_slider"), 0)
        dpg.configure_item(self.tag_with_namespace("frame_slider"), enabled=self.get_number_of_valid_frames() > 0)
        dpg.set_value(
            self.tag_with_namespace("images_folder_path"), app_data["file_path_name"]
        )
        if self.get_number_of_valid_frames() > 0:
            self.load_frame(self.frame_index)

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

    def _update_frame_and_slider(self, new_frame_index):
        self.frame_index = new_frame_index
        self.set_slider_to_frame_index(self.frame_index)
        self.load_frame(self.frame_index)

    def on_step_forward(self):
        self.on_pause()
        valid_indices = self.get_valid_frame_indices()
        if not valid_indices: return
        cur_idx = self.get_slider_index_from_frame_index(self.frame_index)
        if cur_idx + 1 < len(valid_indices):
            self._update_frame_and_slider(valid_indices[cur_idx + 1])

    def on_step_back(self):
        self.on_pause()
        valid_indices = self.get_valid_frame_indices()
        if not valid_indices: return
        cur_idx = self.get_slider_index_from_frame_index(self.frame_index)
        if cur_idx - 1 >= 0:
            self._update_frame_and_slider(valid_indices[cur_idx - 1])

    def on_jump_n_frames_fwd(self):
        self.on_pause()
        valid_indices = self.get_valid_frame_indices()
        if not valid_indices: return
        cur_idx = self.get_slider_index_from_frame_index(self.frame_index)
        jump_amount = dpg.get_value(self.tag_with_namespace("n_frames_jump"))
        target_slider_idx = min(cur_idx + jump_amount, len(valid_indices) - 1)
        self._update_frame_and_slider(valid_indices[target_slider_idx])

    def on_jump_n_frames_bck(self):
        self.on_pause()
        valid_indices = self.get_valid_frame_indices()
        if not valid_indices: return
        cur_idx = self.get_slider_index_from_frame_index(self.frame_index)
        jump_amount = dpg.get_value(self.tag_with_namespace("n_frames_jump"))
        target_slider_idx = max(cur_idx - jump_amount, 0)
        self._update_frame_and_slider(valid_indices[target_slider_idx])

    def on_playback_speed_up(self):
        if self.frame_delay > 0.001: # Avoid division by zero or too small delay
            self.frame_delay *= 0.7

    def on_playback_speed_reset(self):
        self.frame_delay = self.FRAME_DELAY_DEFAULT

    def set_slider_to_frame_index(self, frame_index):
        slider_index = self.get_slider_index_from_frame_index(frame_index)
        dpg.set_value(self.tag_with_namespace("frame_slider"), slider_index)

    def on_remove_frame(self):
        if (
            self.frame_index not in self.images_idx_to_skip
            and self.get_number_of_valid_frames() > 1 # Ensure there's at least one frame left after removal
        ):
            current_slider_idx_before_remove = self.get_slider_index_from_frame_index(self.frame_index)
            self.images_idx_to_skip.append(self.frame_index)
            self.images_idx_to_skip.sort() # Keep it sorted for easier debugging / consistency

            valid_indices_after_remove = self.get_valid_frame_indices()

            if not valid_indices_after_remove: # Should not happen due to check above
                dpg.configure_item(self.tag_with_namespace("frame_slider"), max_value=0, enabled=False)
                # Optionally clear texture or show placeholder
                print("All frames have been removed or skipped.")
                return

            dpg.configure_item(
                self.tag_with_namespace("frame_slider"),
                max_value=len(valid_indices_after_remove) - 1,
            )

            # Decide new frame index:
            # Try to stay at the same slider position, or the last available if current_slider_idx_before_remove is too high
            new_slider_idx = min(current_slider_idx_before_remove, len(valid_indices_after_remove) - 1)
            self.frame_index = valid_indices_after_remove[new_slider_idx]

            self.set_slider_to_frame_index(self.frame_index) # This will update the slider to new_slider_idx
            self.load_frame(self.frame_index)

            print("Currently skipped frames: ", self.images_idx_to_skip)

    def on_undo_last_frame_remove(self):
        if len(self.images_idx_to_skip) > 0:
            # Assuming images_idx_to_skip was sorted, the last one added might not be the largest.
            # For a true "undo last remove", we'd need to store the order of removal.
            # For simplicity, this will remove the largest skipped index if not sorted,
            # or the last appended if sorted (which it is now).
            self.images_idx_to_skip.pop() # Removes the last element
            dpg.configure_item(
                self.tag_with_namespace("frame_slider"),
                max_value=self.get_number_of_valid_frames() - 1 if self.get_number_of_valid_frames() > 0 else 0,
            )
            # Refresh current frame and slider, in case the current frame_index became valid again
            self.set_slider_to_frame_index(self.frame_index)
            self.load_frame(self.frame_index) # Reload, as it might have been a skipped frame
            print("Currently skipped frames: ", self.images_idx_to_skip)

    def on_skip_start(self):
        self.skip_start_idx = self.frame_index
        dpg.configure_item(self.tag_with_namespace("skip_frames_finish"), enabled=True)
        dpg.configure_item(self.tag_with_namespace("reset_skip_frames"), enabled=True)
        print(f"Skip range started at frame: {self.skip_start_idx}")

    def on_reset_skip(self):
        self.skip_start_idx = -1
        dpg.configure_item(self.tag_with_namespace("skip_frames_finish"), enabled=False)
        dpg.configure_item(self.tag_with_namespace("reset_skip_frames"), enabled=False)
        print("Skip range reset.")

    def on_skip_finish(self):
        if self.skip_start_idx == -1:
            return

        skip_finish_idx = self.frame_index
        start_original_idx = min(self.skip_start_idx, skip_finish_idx)
        end_original_idx = max(self.skip_start_idx, skip_finish_idx)

        # Range of frames to skip (original indices)
        indices_to_add_to_skip = list(range(start_original_idx, end_original_idx + 1))
        print(f"Attempting to skip frames from original index {start_original_idx} to {end_original_idx}")

        # Add to skip list (no duplicates) and sort
        for idx in indices_to_add_to_skip:
            if idx not in self.images_idx_to_skip:
                self.images_idx_to_skip.append(idx)
        self.images_idx_to_skip.sort()

        self.skip_start_idx = -1 # Reset skip range start

        valid_indices_after_skip = self.get_valid_frame_indices()
        if not valid_indices_after_skip:
            dpg.configure_item(self.tag_with_namespace("frame_slider"), max_value=0, enabled=False)
            # Optionally clear texture or show placeholder
            print("All frames have been skipped.")
            return

        # Determine the new frame_index: try to go to the frame immediately after the skipped range
        next_frame_candidates = [i for i in valid_indices_after_skip if i > end_original_idx]
        if next_frame_candidates:
            self.frame_index = next_frame_candidates[0]
        else: # If no valid frames after the skipped range, go to the last valid frame available
            self.frame_index = valid_indices_after_skip[-1]

        dpg.configure_item(self.tag_with_namespace("frame_slider"), max_value=len(valid_indices_after_skip) - 1)
        self._update_frame_and_slider(self.frame_index) # This will load frame and set slider

        dpg.configure_item(self.tag_with_namespace("skip_frames_finish"), enabled=False)
        dpg.configure_item(self.tag_with_namespace("reset_skip_frames"), enabled=False)
        print("Currently skipped frames: ", self.images_idx_to_skip)