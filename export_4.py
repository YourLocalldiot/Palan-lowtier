"""Export a trained Keras model to a self-contained TFLite bundle for the Streamlit app."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import tensorflow as tf

from geo_utils import load_centroids

CENTROIDS_PATH = Path("cell_centroids.json")
COUNTRY_ACCURACY_PATH = Path("country_accuracy.json")
EXPORT_DIR = Path("export")
DEFAULT_MODEL_PATH = Path("checkpoints/best.keras")
IMAGE_SIZE = (224, 224)


def convert_to_tflite(model: tf.keras.Model, quantize: bool) -> bytes:
	converter = tf.lite.TFLiteConverter.from_keras_model(model)
	if quantize:
		converter.optimizations = [tf.lite.Optimize.DEFAULT]
	return converter.convert()


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
	parser.add_argument("--normalization", choices=("minus_one_one", "imagenet"), default="minus_one_one")
	parser.add_argument("--no-quantize", dest="quantize", action="store_false", default=True)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	if not args.model_path.exists():
		raise FileNotFoundError(f"{args.model_path} not found; run train_2.py first")
	if not CENTROIDS_PATH.exists():
		raise FileNotFoundError(f"{CENTROIDS_PATH} not found; run data_pipeline_1.py first")

	EXPORT_DIR.mkdir(parents=True, exist_ok=True)

	model = tf.keras.models.load_model(args.model_path)
	tflite_model = convert_to_tflite(model, args.quantize)

	model_path = EXPORT_DIR / "model.tflite"
	model_path.write_bytes(tflite_model)
	print(f"Exported TFLite model to {model_path} ({len(tflite_model) / 1e6:.2f} MB)")

	shutil.copyfile(CENTROIDS_PATH, EXPORT_DIR / "cell_centroids.json")
	print(f"Copied {CENTROIDS_PATH} to {EXPORT_DIR / 'cell_centroids.json'}")

	if COUNTRY_ACCURACY_PATH.exists():
		shutil.copyfile(COUNTRY_ACCURACY_PATH, EXPORT_DIR / "country_accuracy.json")
		print(f"Copied {COUNTRY_ACCURACY_PATH} to {EXPORT_DIR / 'country_accuracy.json'}")
	else:
		print(f"No {COUNTRY_ACCURACY_PATH} found; run evaluate_3.py to generate it for the Model Stats page.")

	config = {
		"image_size": list(IMAGE_SIZE),
		"normalization": args.normalization,
		"num_classes": len(load_centroids(CENTROIDS_PATH)),
	}
	config_path = EXPORT_DIR / "config.json"
	config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
	print(f"Wrote {config_path}")


if __name__ == "__main__":
	main()
