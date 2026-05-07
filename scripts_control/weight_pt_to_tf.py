import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import torch
import numpy as np
import tensorflow as tf
from dataclasses import dataclass

from utils_ctrl_lya_tf import ControllerTF, LyapunovTF


@dataclass
class Config:

    pytorch_file = "weights/ctrl_lya.pt"

    controller_out = "weights/ctrl.tflite"
    lyapunov_out = "weights/lya.tflite"

    tf_float = tf.float32  
    width = 320
    height = 240

    target_pose = np.array(
        [0.0, -4.0, 0.0, 1.57, 0.0, 0.0],
        dtype=np.float32
    )


def get_all_layers(model):

    layers = []

    for layer in model.layers:

        layers.append(layer)

        if hasattr(layer, "layers") and len(layer.layers) > 0:
            layers.extend(get_all_layers(layer))

    return layers


def _set_tf_layer_weights_from_pt(layer, pt_state_dict):

    if isinstance(layer, tf.keras.layers.Dense):

        w_key = layer.name + ".weight"
        b_key = layer.name + ".bias"

        if w_key in pt_state_dict:

            w = pt_state_dict[w_key].cpu().numpy().T

            b = pt_state_dict[b_key].cpu().numpy()

            layer.set_weights([w, b])

            print(f"Loaded Dense: {w_key}")

        else:
            print(f"Missing Dense: {w_key}")

    elif isinstance(layer, tf.keras.layers.Conv2D):

        w_key = layer.name + ".weight"
        b_key = layer.name + ".bias"

        if w_key in pt_state_dict:

            w = (
                pt_state_dict[w_key]
                .cpu()
                .numpy()
                .transpose(2, 3, 1, 0)
            )

            b = pt_state_dict[b_key].cpu().numpy()

            layer.set_weights([w, b])

            print(f"Loaded Conv2D: {w_key}")

        else:
            print(f"Missing Conv2D: {w_key}")

    elif isinstance(layer, tf.keras.layers.BatchNormalization):

        gamma_key = layer.name + ".weight"
        beta_key = layer.name + ".bias"
        mean_key = layer.name + ".running_mean"
        var_key = layer.name + ".running_var"

        required = [
            gamma_key,
            beta_key,
            mean_key,
            var_key
        ]

        if all(k in pt_state_dict for k in required):

            gamma = pt_state_dict[gamma_key].cpu().numpy()
            beta = pt_state_dict[beta_key].cpu().numpy()
            mean = pt_state_dict[mean_key].cpu().numpy()
            var = pt_state_dict[var_key].cpu().numpy()

            layer.set_weights([
                gamma,
                beta,
                mean,
                var
            ])

            print(f"Loaded BatchNorm: {layer.name}")

        else:
            print(f"Missing BatchNorm: {layer.name}")


def export_tflite(model, output_path):

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    print(f"Saved TFLite model: {output_path}")


def convert_pytorch_weights_to_tensorflow():

    cfg = Config()

    checkpoint = torch.load(
        cfg.pytorch_file,
        map_location="cpu"
    )

    controller_state = checkpoint["controller"]
    lyapunov_state = checkpoint["lyapunov"]

    print("\n==== CONTROLLER KEYS ====\n")

    for k in controller_state.keys():
        print(k)

    print("\n==== LYAPUNOV KEYS ====\n")

    for k in lyapunov_state.keys():
        print(k)

    print("\n==== END KEYS ====\n")

    # Create TF models
    controller_tf = ControllerTF()

    lyapunov_tf = LyapunovTF()

    # Build models
    dummy_controller_input = tf.zeros(
        (1, cfg.height, cfg.width, 3),
        dtype=cfg.tf_float
    )

    _ = controller_tf(
        dummy_controller_input,
        training=False
    )

    dummy_lya_input = tf.zeros(
        (1, 6),
        dtype=cfg.tf_float
    )

    dummy_target_pose = tf.convert_to_tensor(
        cfg.target_pose[None, :],
        dtype=cfg.tf_float
    )

    _ = lyapunov_tf(
        dummy_lya_input,
        dummy_target_pose,
        training=False
    )

    # Load controller weights
    for layer in get_all_layers(controller_tf):
        _set_tf_layer_weights_from_pt(
            layer,
            controller_state
        )

    # Load Lyapunov weights
    for layer in get_all_layers(lyapunov_tf):
        _set_tf_layer_weights_from_pt(
            layer,
            lyapunov_state
        )

    # Export TFLite
    export_tflite(
        controller_tf,
        cfg.controller_out
    )

    export_tflite(
        lyapunov_tf,
        cfg.lyapunov_out
    )


if __name__ == "__main__":

    convert_pytorch_weights_to_tensorflow()