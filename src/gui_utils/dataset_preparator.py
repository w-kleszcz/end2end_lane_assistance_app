import os
import yaml
import dearpygui.dearpygui as dpg
from .player import Player # Assuming Player is in the same directory

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