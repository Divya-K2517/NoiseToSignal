
# The JSON schema
# {
#   "model_name":      "fashion_mnist",
#   "meta_accuracy":   0.838,         // optional, shown in the UI
#   "meta_loss":       0.461,         // optional
#   "meta_epochs":     20,            // optional
#   "layer_0_weights": [[...], ...],  // shape (784, 128)
#   "layer_0_biases":  [[...]],       // shape (1, 128)
#   "layer_1_weights": [[...], ...],  // shape (128, 64)
#   "layer_1_biases":  [[...]],       // shape (1, 64)
#   "layer_2_weights": [[...], ...],  // shape (64, 10)
#   "layer_2_biases":  [[...]],       // shape (1, 10)
# }


import json
import numpy as np

# ── option A: export directly after training ─────────────────────────────────
# If you run this at the bottom of Fashion-MNIST.py (after training), just
# import the already-trained layer objects directly.  Example:
#
#   from Fashion-MNIST import dense1, dense2, dense3, test_acc, test_loss, epochs
#   layers    = [dense1, dense2, dense3]
#   meta      = {"accuracy": float(test_acc), "loss": float(test_loss), "epochs": epochs}
#
# ── option B: load from a saved .npz file (recommended for a clean workflow) ─
# Train once, save with save_model(), then run this script any time.

from network import load_model

NPZ_PATH  = "models/fashion_mnist.npz"   # path to the file saved by save_model()
OUT_PATH  = "fashion_mnist_weights.json"

def export(layers, meta, out_path):
    payload = {
        "model_name": meta.get("model_name", "fashion_mnist"),
    }

    # pull any meta_ keys out of the dict and store them top-level
    for key, val in meta.items():
        if key.startswith("meta_"):
            payload[key] = val          # e.g. "meta_accuracy", "meta_loss"

    for i, layer in enumerate(layers):
        # tolist() converts numpy arrays to plain Python lists so json.dumps works
        payload[f"layer_{i}_weights"] = layer.weights.tolist()
        payload[f"layer_{i}_biases"]  = layer.biases.tolist()

    with open(out_path, "w") as f:
        json.dump(payload, f, separators=(",", ":"))   # compact, no extra spaces

    size_kb = len(json.dumps(payload, separators=(",", ":")).encode()) / 1024
    print(f"Exported {len(layers)} layers → {out_path}  ({size_kb:.0f} KB)")
    for i, layer in enumerate(layers):
        print(f"  layer {i}: weights {layer.weights.shape}  biases {layer.biases.shape}")


if __name__ == "__main__":
    layers, meta = load_model(NPZ_PATH)
    export(layers, meta, OUT_PATH)