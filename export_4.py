"""Export a trained Keras model to a self-contained TFLite bundle for the Streamlit app."""

# =============================================================================
# export_4.py - STEP 4 of 5: package the model for the web app
# -----------------------------------------------------------------------------
# Input : checkpoints/best.keras   (from train_2.py)
#         cell_centroids.json      (from data_pipeline_1.py)
#         country_accuracy.json    (from evaluate_3.py - optional)
# Output: the export/ folder, which is everything app.py needs:
#           export/model.tflite           the compressed model
#           export/cell_centroids.json    geocell number -> lat/lon lookup
#           export/country_accuracy.json  numbers for the Model Stats page
#           export/config.json            settings the app must match
#
# WHY TENSORFLOW LITE?
#   The full Keras model (.keras) is ~26 MB and needs the whole training-ready
#   TensorFlow setup. TensorFlow Lite (.tflite) is a slimmed-down format made
#   for running a finished model quickly - on phones, or here, on a free web
#   server with no GPU. With quantization it shrinks to under 3 MB.
#
# The export/ folder is committed to git on purpose: Streamlit Cloud builds the
# live app from the GitHub repo, so the model files must be in the repo. (The
# much larger data/ and checkpoints/ folders are NOT committed.)
# =============================================================================

from __future__ import annotations

import argparse            # read command-line flags
import json                # write config.json
import shutil              # copy files
from pathlib import Path   # file paths

import tensorflow as tf    # load the Keras model and convert it to TFLite

from geo_utils import load_centroids

CENTROIDS_PATH = Path("cell_centroids.json")
COUNTRY_ACCURACY_PATH = Path("country_accuracy.json")
# Everything the Streamlit app reads lives in this one folder.
EXPORT_DIR = Path("export")
DEFAULT_MODEL_PATH = Path("checkpoints/best.keras")
# Must match the image size the model was trained on (see data_pipeline_1.py).
IMAGE_SIZE = (224, 224)


def convert_to_tflite(model: tf.keras.Model, quantize: bool) -> bytes:
	# Convert the Keras model into TensorFlow Lite format (returned as bytes).
	converter = tf.lite.TFLiteConverter.from_keras_model(model)
	if quantize:
		# QUANTIZATION: store the model's weights as 8-bit integers instead of
		# 32-bit decimal numbers. That makes the file roughly 4x smaller and
		# faster to run, at the cost of a very small loss of precision - usually
		# a negligible difference in accuracy.
		converter.optimizations = [tf.lite.Optimize.DEFAULT]
	return converter.convert()


def parse_args() -> argparse.Namespace:
	# Optional command-line flags, e.g. python export_4.py --no-quantize
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
	# Recorded in config.json so the app prepares uploaded images exactly the
	# same way the training images were prepared. Must match training.
	parser.add_argument("--normalization", choices=("minus_one_one", "imagenet"), default="minus_one_one")
	# Quantization is ON by default; --no-quantize turns it off.
	parser.add_argument("--no-quantize", dest="quantize", action="store_false", default=True)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	# Make sure the earlier steps have produced what we need.
	if not args.model_path.exists():
		raise FileNotFoundError(f"{args.model_path} not found; run train_2.py first")
	if not CENTROIDS_PATH.exists():
		raise FileNotFoundError(f"{CENTROIDS_PATH} not found; run data_pipeline_1.py first")

	# Create export/ if it doesn't exist yet.
	EXPORT_DIR.mkdir(parents=True, exist_ok=True)

	# 1. Load the trained model and convert it to TFLite.
	model = tf.keras.models.load_model(args.model_path)
	tflite_model = convert_to_tflite(model, args.quantize)

	# 2. Save the converted model.
	model_path = EXPORT_DIR / "model.tflite"
	model_path.write_bytes(tflite_model)
	print(f"Exported TFLite model to {model_path} ({len(tflite_model) / 1e6:.2f} MB)")

	# 3. Copy the geocell centres. The model only outputs geocell NUMBERS;
	#    this file is how the app turns "geocell 47" into a real lat/lon.
	#    It must come from the same data_pipeline_1.py run the model was trained
	#    on, or the numbers will point to the wrong places.
	shutil.copyfile(CENTROIDS_PATH, EXPORT_DIR / "cell_centroids.json")
	print(f"Copied {CENTROIDS_PATH} to {EXPORT_DIR / 'cell_centroids.json'}")

	# 4. Copy the per-country accuracy for the Model Stats page, if evaluation
	#    has been run. It's optional, so a missing file is a warning, not an error.
	if COUNTRY_ACCURACY_PATH.exists():
		shutil.copyfile(COUNTRY_ACCURACY_PATH, EXPORT_DIR / "country_accuracy.json")
		print(f"Copied {COUNTRY_ACCURACY_PATH} to {EXPORT_DIR / 'country_accuracy.json'}")
	else:
		print(f"No {COUNTRY_ACCURACY_PATH} found; run evaluate_3.py to generate it for the Model Stats page.")

	# 5. Write the settings the app needs to prepare uploaded photos correctly,
	#    so nothing has to be hard-coded (and possibly mismatched) in app.py.
	config = {
		"image_size": list(IMAGE_SIZE),                          # resize photos to this
		"normalization": args.normalization,                    # scale pixels this way
		"num_classes": len(load_centroids(CENTROIDS_PATH)),     # number of geocells
	}
	config_path = EXPORT_DIR / "config.json"
	config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
	print(f"Wrote {config_path}")


# Run main() only when this file is executed directly (python export_4.py).
if __name__ == "__main__":
	main()
