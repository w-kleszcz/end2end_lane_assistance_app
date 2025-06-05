import dearpygui.dearpygui as dpg
from .player import Player  # Assuming Player is in the same directory


class ModelEvaluator(Player):
    def __init__(
        self, texture_id, image_folder="", image_annotations_file="", namespace=""
    ):
        super().__init__(texture_id, image_folder, namespace)

        self.dataset_yaml = None
        self.image_annotations_file = image_annotations_file

    def set_dataset_yaml(self, sender, app_data):
        selected_file_path = app_data["file_path_name"]
        self.dataset_yaml = selected_file_path
        self.setup_player_from_yaml_data(selected_file_path)
        dpg.set_value(
            self.tag_with_namespace("dataset_yaml_file_path"),
            app_data["file_path_name"],
        )
