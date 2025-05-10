import os
import cv2
import glob
import threading
import time
import numpy as np
import dearpygui.dearpygui as dpg
from natsort import natsorted


class Player:
    FRAME_DELAY_DEFAULT = 0.033 # ~30 FPS

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


# ---------- Create texture buffer ----------
dpg.create_context()
dpg.create_viewport(title="End2end Lane Assistance App", width=800, height=600)

with dpg.texture_registry():
    texture_id = dpg.generate_uuid()
    dummy_image = np.zeros((480, 640, 4), dtype=np.float32)
    dpg.add_dynamic_texture(640, 480, dummy_image.flatten(), tag=texture_id)


player = Player("data/raw/07012018/data", texture_id)


# ---------- GUI Layout ----------
with dpg.window(label="Player", width=700, height=550):
    dpg.add_image(texture_id)

    with dpg.group(horizontal=True):
        dpg.add_button(label="Play", callback=player.on_play)
        dpg.add_button(label="Pause", callback=player.on_pause)
        dpg.add_button(label="Step Forward", callback=player.on_step_forward)
        dpg.add_button(label="Step Back", callback=player.on_step_back)
        dpg.add_button(label="Speed Up", callback=player.on_playback_speed_up)
        dpg.add_button(label="Speed Reset", callback=player.on_playback_speed_reset)


# ---------- Launch ----------
dpg.setup_dearpygui()
dpg.show_viewport()
player.load_frame(0)  # Load the first frame initially
dpg.start_dearpygui()
dpg.destroy_context()
