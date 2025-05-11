import keras
import config


def build_pilotnet_model(input_shape):
    """
    Builds the PilotNet model using the Keras 3.0 API.
    Assumes 'channels_first' data format (Channels, Height, Width)
    for the PyTorch backend if input_shape is (C, H, W).
    """
    model = keras.Sequential(name="PilotNet_Keras3_PyTorchBackend")
    model.add(keras.layers.Input(shape=input_shape, name="InputLayer"))

    model.add(
        keras.layers.Conv2D(
            24,
            (5, 5),
            strides=(2, 2),
            activation="relu",
            data_format="channels_first",
            name="Conv1",
        )
    )
    model.add(
        keras.layers.Conv2D(
            36,
            (5, 5),
            strides=(2, 2),
            activation="relu",
            data_format="channels_first",
            name="Conv2",
        )
    )
    model.add(
        keras.layers.Conv2D(
            48,
            (5, 5),
            strides=(2, 2),
            activation="relu",
            data_format="channels_first",
            name="Conv3",
        )
    )
    model.add(
        keras.layers.Conv2D(
            64, (3, 3), activation="relu", data_format="channels_first", name="Conv4"
        )
    )
    model.add(
        keras.layers.Conv2D(
            64, (3, 3), activation="relu", data_format="channels_first", name="Conv5"
        )
    )

    model.add(
        keras.layers.Flatten(name="FlattenLayer")
    )  # Flatten automatically handles data_format

    # Fully connected layers
    model.add(keras.layers.Dense(100, activation="relu", name="Dense1"))
    model.add(keras.layers.Dense(50, activation="relu", name="Dense2"))
    model.add(keras.layers.Dense(10, activation="relu", name="Dense3"))
    model.add(
        keras.layers.Dense(1, name="Output_SteeringAngle")
    )  # Single output for steering angle

    return model
