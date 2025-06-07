import os
import numpy as np
import dearpygui.dearpygui as dpg

from gui_utils.player import Player
from gui_utils.dataset_preparator import DatasetPreparator
from gui_utils.model_evaluator import ModelEvaluator
from gui_utils.training_ui import (
    create_training_tab_content,
    update_training_log_display,
)

APP_WINDOW_WIDTH = 660
APP_WINDOW_HEIGHT = 830
UI_INPUT_WIDTH_LONG = 550

# ---------- Create texture buffer ----------
dpg.create_context()

dpg.create_viewport(
    title="End2end Lane Assistance App",
    width=APP_WINDOW_WIDTH,
    height=APP_WINDOW_HEIGHT,  # Adjusted for potential scrollbar with many tabs
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

    deployed_player_texture_id = dpg.generate_uuid()
    dpg.add_dynamic_texture(
        640, 480, dummy_image.flatten(), tag=deployed_player_texture_id
    )

# ---------- GUI Layout ----------
with dpg.window(
    label="End2end Lane Assistance App",
    width=APP_WINDOW_WIDTH,
    height=APP_WINDOW_HEIGHT,
    tag="app_primary_window",  # Added tag for error popup centering
):
    with dpg.tab_bar(tag="main_tab_bar") as main_tab_bar:

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
                    label="Save dataset metadata",
                    callback=dataset_preparator.on_save_dataset,
                )

        # The Model Training tab is now created by create_training_tab_content
        create_training_tab_content(parent_tab_id=main_tab_bar)

        with dpg.tab(label="Model Evaluation"):

            model_evaluator = ModelEvaluator(
                texture_id=evaluator_player_texture_id, namespace="evaluator"
            )

            dpg.add_text("Dataset specs file:")
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Browse",
                    callback=lambda: dpg.show_item(
                        "evaluator::dataset_specs_file_dialog"
                    ),
                )
                dpg.add_input_text(
                    tag="evaluator::dataset_yaml_file_path",
                    width=UI_INPUT_WIDTH_LONG,
                    readonly=True,
                )  # Textbox for file path

            dpg.add_text("Model file:")
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Browse",
                    callback=lambda: dpg.show_item("evaluator::model_file_dialog"),
                )
                dpg.add_input_text(
                    tag="evaluator::model_file_path",
                    width=UI_INPUT_WIDTH_LONG,
                    readonly=True,
                )  # Textbox for file path

            # File Dialog (for selecting a single file)
            with dpg.file_dialog(
                directory_selector=False,
                show=False,
                tag="evaluator::dataset_specs_file_dialog",
                callback=model_evaluator.set_dataset_yaml,
                width=500,
                height=400,
            ):
                dpg.add_file_extension(".yaml")

            # File Dialog (for selecting a single file)
            with dpg.file_dialog(
                directory_selector=False,
                show=False,
                tag="evaluator::model_file_dialog",
                callback=model_evaluator.set_model,
                width=500,
                height=400,
            ):
                dpg.add_file_extension(".pth")

            dpg.add_image(evaluator_player_texture_id)

            dpg.add_slider_int(
                tag="evaluator::frame_slider",
                label="",
                min_value=0,
                max_value=0,
                default_value=0,
                width=640,
                callback=model_evaluator.on_player_slider_change,
                enabled=True,
            )

            with dpg.group(horizontal=True):
                dpg.add_text("MEASURED ANGLE=")
                dpg.add_input_float(
                    tag="evaluator::measured_angle",
                    width=100,
                    default_value=0.0,
                    readonly=True,
                )
                dpg.add_text("PREDICTED ANGLE=")
                dpg.add_input_float(
                    tag="evaluator::predicted_angle",
                    width=100,
                    default_value=0.0,
                    readonly=True,
                )
                dpg.add_text("ERROR=")
                dpg.add_input_float(
                    tag="evaluator::angle_error",
                    width=100,
                    default_value=0.0,
                    readonly=True,
                )

            with dpg.group(horizontal=True):
                dpg.add_button(label="Play", callback=model_evaluator.on_play)
                dpg.add_button(label="Pause", callback=model_evaluator.on_pause)
                dpg.add_button(label="Step Back", callback=model_evaluator.on_step_back)
                dpg.add_button(
                    label="Step Forward", callback=model_evaluator.on_step_forward
                )
                dpg.add_button(
                    label="Speed Up", callback=model_evaluator.on_playback_speed_up
                )
                dpg.add_button(
                    label="Speed Reset",
                    callback=model_evaluator.on_playback_speed_reset,
                )

            with dpg.group(horizontal=True):
                dpg.add_text("N =")
                dpg.add_input_int(
                    tag="evaluator::n_frames_jump", width=100, default_value=100
                )
                dpg.add_button(
                    label="Jump N Frames Back",
                    tag="evaluator::jump_n_frames_bck",
                    callback=model_evaluator.on_jump_n_frames_bck,
                )
                dpg.add_button(
                    label="Jump N Frames Forward",
                    tag="evaluator::jump_n_frames_fwd",
                    callback=model_evaluator.on_jump_n_frames_fwd,
                )

        with dpg.tab(label="Model Deployment"):

            deployed_model_player = Player(
                texture_id=deployed_player_texture_id, namespace="deployed"
            )

            dpg.add_text("Images folder:")
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Browse",
                    callback=lambda: dpg.show_item("deployed::images_folder_dialog"),
                )
                dpg.add_input_text(
                    tag="deployed::images_folder_path",
                    width=UI_INPUT_WIDTH_LONG,
                    readonly=True,
                )  # Textbox for file path

            dpg.add_text("Model file:")
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Browse",
                    callback=lambda: dpg.show_item("deployed::model_file_dialog"),
                )
                dpg.add_input_text(
                    tag="deployed::model_file_path",
                    width=UI_INPUT_WIDTH_LONG,
                    readonly=True,
                )  # Textbox for file path

            # File Dialog (for selecting a single file)
            with dpg.file_dialog(
                directory_selector=True,
                show=False,
                tag="deployed::images_folder_dialog",
                callback=deployed_model_player.on_folder_selected,
                width=500,
                height=400,
            ):
                dpg.add_file_extension(".*")

            # File Dialog (for selecting a single file)
            with dpg.file_dialog(
                directory_selector=False,
                show=False,
                tag="deployed::model_file_dialog",
                callback=deployed_model_player.set_model,
                width=500,
                height=400,
            ):
                dpg.add_file_extension(".pth")

            dpg.add_image(deployed_player_texture_id)

            dpg.add_slider_int(
                tag="deployed::frame_slider",
                label="",
                min_value=0,
                max_value=0,
                default_value=0,
                width=640,
                callback=deployed_model_player.on_player_slider_change,
                enabled=True,
            )

            with dpg.group(horizontal=True):
                dpg.add_text("PREDICTED ANGLE=")
                dpg.add_input_float(
                    tag="deployed::predicted_angle",
                    width=100,
                    default_value=0.0,
                    readonly=True,
                )

            with dpg.group(horizontal=True):
                dpg.add_button(label="Play", callback=deployed_model_player.on_play)
                dpg.add_button(label="Pause", callback=deployed_model_player.on_pause)
                dpg.add_button(
                    label="Step Back", callback=deployed_model_player.on_step_back
                )
                dpg.add_button(
                    label="Step Forward", callback=deployed_model_player.on_step_forward
                )
                dpg.add_button(
                    label="Speed Up",
                    callback=deployed_model_player.on_playback_speed_up,
                )
                dpg.add_button(
                    label="Speed Reset",
                    callback=deployed_model_player.on_playback_speed_reset,
                )

            with dpg.group(horizontal=True):
                dpg.add_text("N =")
                dpg.add_input_int(
                    tag="deployed::n_frames_jump", width=100, default_value=100
                )
                dpg.add_button(
                    label="Jump N Frames Back",
                    tag="deployed::jump_n_frames_bck",
                    callback=deployed_model_player.on_jump_n_frames_bck,
                )
                dpg.add_button(
                    label="Jump N Frames Forward",
                    tag="deployed::jump_n_frames_fwd",
                    callback=deployed_model_player.on_jump_n_frames_fwd,
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
