import os
import cv2
import glob
import threading
import time
import numpy as np
import yaml
import dearpygui.dearpygui as dpg
from natsort import natsorted
from PIL import Image
from pathlib import Path
import torch
from torchcam.methods import SmoothGradCAMpp
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
from model.model import PilotNetPyTorch

from model.data_loader import (
    create_dataloaders_from_yaml,
    get_data_transforms,
    parse_annotations_file,
)


def make_white_background_transparent(image: Image.Image, threshold=240) -> Image.Image:
    image = image.convert("RGBA")
    data = image.getdata()

    new_data = []
    for item in data:
        # Detect white or near-white pixels
        if item[0] >= threshold and item[1] >= threshold and item[2] >= threshold:
            new_data.append((255, 255, 255, 0))  # Fully transparent
        else:
            new_data.append(item)

    image.putdata(new_data)
    return image


def set_image_opacity(image: Image.Image, opacity: float) -> Image.Image:
    """
    Sets the opacity of an RGBA image. `opacity` should be between 0.0 (fully transparent) and 1.0 (fully opaque).
    """
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    alpha = image.split()[3]
    alpha = alpha.point(lambda p: int(p * opacity))
    image.putalpha(alpha)
    return image


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

        self.model = None
        self.image_annotations_file = None
        self.img_to_steering = None
        self.device = None
        self.cam_extractor = None

        self.driving_wheel_size = (100, 100)
        self.driving_wheel_meas = Image.open("src/assets/driving_wheel.png").convert(
            "RGBA"
        )
        self.driving_wheel_meas = make_white_background_transparent(
            self.driving_wheel_meas
        )
        self.driving_wheel_meas = self.driving_wheel_meas.resize(
            self.driving_wheel_size
        )
        self.driving_wheel_pred = Image.open(
            "src/assets/driving_wheel_green.png"
        ).convert("RGBA")
        self.driving_wheel_pred = make_white_background_transparent(
            self.driving_wheel_pred
        )
        self.driving_wheel_pred = self.driving_wheel_pred.resize(
            self.driving_wheel_size
        )
        self.driving_wheel_pred = set_image_opacity(self.driving_wheel_pred, 0.6)

        with dpg.theme() as self.red_theme:
            with dpg.theme_component(dpg.mvInputFloat):
                dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 0, 0, 255))  # Red

        # Default text theme (inherits app's default)
        with dpg.theme() as self.default_theme:
            with dpg.theme_component(dpg.mvInputFloat):
                dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 255, 255, 255))

    def setup_player_from_yaml_data(self, yaml_file_path):
        with open(yaml_file_path, "r") as file:
            config = yaml.safe_load(file)

        all_image_paths = natsorted(
            glob.glob(os.path.join(config["images_dir"], "*.jpg"))
        )
        original_skip_indices = set(config["indices_to_skip"])
        test_idx_start = config["test_set_idx_start"]
        test_idx_end = config["test_set_idx_end"]

        self.image_annotations_file = config["annotations_file"]
        self.img_to_steering = parse_annotations_file(self.image_annotations_file)

        # Filter image paths
        self.image_paths = all_image_paths[test_idx_start:test_idx_end]

        # Re-index skip indices that fall within the kept range
        self.indexes_to_skip = [
            i - test_idx_start
            for i in original_skip_indices
            if test_idx_start <= i <= test_idx_end
        ]

        dpg.configure_item(
            self.tag_with_namespace("frame_slider"),
            max_value=len(self.get_valid_frame_indices()) - 1,
        )
        dpg.configure_item(self.tag_with_namespace("frame_slider"), enabled=True)

    def set_model(self, sender, app_data):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PilotNetPyTorch()
        self.model.load_state_dict(
            torch.load(app_data["file_path_name"], map_location=self.device)
        )
        self.model.eval()
        self.cam_extractor = SmoothGradCAMpp(self.model, target_layer="conv5")
        self.model.to(self.device)

        dpg.set_value(
            self.tag_with_namespace("model_file_path"), app_data["file_path_name"]
        )

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

        if self.model is None:
            dpg.set_value(self.texture_id, img.flatten())
        else:
            orig_image = Image.open(self.image_paths[index]).convert("RGB")
            transform = get_data_transforms()["val"]

            image = transform(orig_image)
            image = image.unsqueeze(0)
            image = image.to(self.device)

            self.model.eval()  # Just to be safe
            with torch.enable_grad():
                output = self.model(image)  # Shape [1, 1] or [1]
                pred = output.item()

            # Rotate wheels
            wheel_pred = self.driving_wheel_pred.rotate(
                -pred, resample=Image.BICUBIC, expand=False
            )

            class_idx = 0
            activation_map = self.cam_extractor(class_idx, output)
            # Get CAM and ensure it's on CPU
            cam = activation_map[0].detach().cpu()

            # Fix dimensionality: add batch and channel dimensions if needed
            if cam.dim() == 2:
                cam = cam.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            elif cam.dim() == 3:
                cam = cam.unsqueeze(0)  # [1, C, H, W]

            target_size = (orig_image.size[1], orig_image.size[0])
            cam_resized = torch.nn.functional.interpolate(
                cam, size=target_size, mode="bilinear", align_corners=False
            )
            cam_resized = cam_resized.squeeze()

            # Convert to RGB heatmap and save
            heatmap = ToPILImage()(cam_resized.expand(3, -1, -1))
            width, height = orig_image.size
            # Compute region from bottom (e.g., 10% to 60% from bottom)
            bottom = height - int(height * 0.10)  # 10% from bottom
            top = height - int(height * 0.60)  # 60% from bottom
            masked_heatmap = Image.new("RGB", (width, height), (0, 0, 0))
            heatmap_cropped = heatmap.crop((0, top, width, bottom))

            # Paste cropped heatmap back into the masked image
            masked_heatmap.paste(heatmap_cropped, (0, top))

            if self.img_to_steering is not None:
                img_name = Path(self.image_paths[index]).name
                img_with_steering = next(
                    (
                        img_steering
                        for img_steering in self.img_to_steering
                        if img_steering[0] == img_name
                    ),
                    None,
                )
                measured = img_with_steering[1] if img_with_steering else None

                if measured is not None:
                    dpg.set_value(self.tag_with_namespace("measured_angle"), measured)
                    dpg.set_value(
                        self.tag_with_namespace("angle_error"), pred - measured
                    )

                    if abs(pred - measured) > 10.0:
                        dpg.bind_item_theme(
                            self.tag_with_namespace("angle_error"), self.red_theme
                        )
                    else:
                        dpg.bind_item_theme(
                            self.tag_with_namespace("angle_error"), self.default_theme
                        )

                    wheel_meas = self.driving_wheel_meas.rotate(
                        -measured, resample=Image.BICUBIC, expand=False
                    )

            dpg.set_value(self.tag_with_namespace("predicted_angle"), pred)

            overlay = Image.blend(orig_image, masked_heatmap, alpha=0.3)
            # overlay.save("overlay_" + img_name)
            overlay = overlay.convert("RGBA")  # Convert to RGBA
            overlay = overlay.resize(
                (640, 480)
            )  # Resize to match DPG texture size if needed

            wheel_y = (
                overlay.height - self.driving_wheel_size[1] - 10
            )  # 10px padding from bottom
            # pred_x = overlay.width - self.driving_wheel_size[0] - 10    # right side
            wheel_x = int((overlay.width - self.driving_wheel_size[0]) / 2)  # left side
            if self.img_to_steering is not None and measured is not None:
                overlay.paste(wheel_meas, (wheel_x, wheel_y), wheel_meas)

            # Paste predicted steering wheel
            overlay.paste(wheel_pred, (wheel_x, wheel_y), wheel_pred)

            overlay_np = np.array(overlay).astype(np.float32) / 255.0  # Normalize
            overlay_flat = overlay_np.flatten()

            dpg.set_value(self.texture_id, overlay_flat)

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
