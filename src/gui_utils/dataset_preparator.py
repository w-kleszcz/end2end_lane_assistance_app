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

    def on_image_annotations_file_selected(self, sender, app_data, user_data):
        print("Selected image annotations file: ", app_data["file_path_name"])
        self.image_annotations_file = app_data["file_path_name"]
        dpg.set_value(self.tag_with_namespace("annotations_file_path"), app_data["file_path_name"])

    def on_test_dataset_start(self):
        self.test_set_idx_start = self.frame_index
        print("Test set start index set to ", self.test_set_idx_start)

    def on_test_dataset_finish(self):
        self.test_set_idx_finish = self.frame_index
        print("Test set finish index set to ", self.test_set_idx_finish)

    def on_save_dataset(self):
        if not self.image_annotations_file:
            self.show_error_popup("Please select image annotations file.")
            return

        if not self.images_folder:
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
            "indices_to_skip": sorted(list(set(self.images_idx_to_skip))), # Ensure sorted unique list
            "test_set_idx_start": self.test_set_idx_start,
            "test_set_idx_end": self.test_set_idx_finish,
        }
        try:
            with open(app_data["file_path_name"], "w") as f:
                yaml.dump(data, f, default_flow_style=False)
            print(f"Dataset metadata saved to {app_data['file_path_name']}")
        except Exception as e:
            self.show_error_popup(f"Error saving file: {e}")
            print(f"Error saving file: {e}")

    def show_error_popup(self, message):
        dpg.set_value(self.tag_with_namespace("error_message_text"), message)
        # Attempt to center the popup if the main window exists and is visible
        try:
            if dpg.does_item_exist("app_primary_window") and dpg.is_item_shown("app_primary_window"):
                viewport_width = dpg.get_item_rect_size("app_primary_window")[0]
                viewport_height = dpg.get_item_rect_size("app_primary_window")[1]
                popup_width = dpg.get_item_width(self.tag_with_namespace("error_popup"))
                popup_height = dpg.get_item_height(self.tag_with_namespace("error_popup"))
                if popup_width and popup_height: # Ensure dimensions are valid
                    pos_x = (viewport_width - popup_width) // 2
                    pos_y = (viewport_height - popup_height) // 2
                    dpg.set_item_pos(self.tag_with_namespace("error_popup"), [pos_x, pos_y])
        except Exception as e:
            print(f"Could not center error popup: {e}") # Non-critical error
            # Fallback position if centering fails
            dpg.set_item_pos(self.tag_with_namespace("error_popup"), [150,150])

        dpg.show_item(self.tag_with_namespace("error_popup"))

    # Inherits frame removal and skipping logic from Player