import dearpygui.dearpygui as dpg
from .player import Player  # Assuming Player is in the same directory
import torch
from model.model import PilotNetPyTorch
from torchcam.methods import SmoothGradCAMpp


class ModelEvaluator(Player):
    def __init__(
        self, texture_id, image_folder="", image_annotations_file="", namespace=""
    ):
        super().__init__(texture_id, image_folder, namespace)

        self.dataset_yaml = None
        self.image_annotations_file = image_annotations_file

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

    def set_dataset_yaml(self, sender, app_data):
        selected_file_path = app_data["file_path_name"]
        self.dataset_yaml = selected_file_path
        self.setup_player_from_yaml_data(selected_file_path)
        dpg.set_value(
            self.tag_with_namespace("dataset_yaml_file_path"),
            app_data["file_path_name"],
        )
