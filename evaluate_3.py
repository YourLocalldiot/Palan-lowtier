"""Evaluate a trained geocell classifier using haversine distance error."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from data_pipeline_1 import CENTROIDS_PATH, build_eval_dataset
from geo_utils import accuracy_at_thresholds, haversine, load_centroids

DEV_TFRECORD = Path("data/processed/dev.tfrecord")
DEFAULT_MODEL_PATH = Path("checkpoints/best.keras")


def evaluate(
	model: tf.keras.Model,
	dataset: tf.data.Dataset,
	centroids: dict[int, tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray]:
	"""Return (haversine_distances_km, top1_cell_correct) across the dataset."""
	num_classes = len(centroids)
	centroid_lat = np.array([centroids[i][0] for i in range(num_classes)])
	centroid_lon = np.array([centroids[i][1] for i in range(num_classes)])

	true_lat_batches, true_lon_batches = [], []
	pred_lat_batches, pred_lon_batches = [], []
	correct_batches = []
	for images, cell_ids, latitudes, longitudes in dataset:
		probabilities = model.predict(images, verbose=0)
		predicted_cells = np.argmax(probabilities, axis=1)
		true_lat_batches.append(latitudes.numpy())
		true_lon_batches.append(longitudes.numpy())
		pred_lat_batches.append(centroid_lat[predicted_cells])
		pred_lon_batches.append(centroid_lon[predicted_cells])
		correct_batches.append(predicted_cells == cell_ids.numpy())

	true_lat = np.concatenate(true_lat_batches)
	true_lon = np.concatenate(true_lon_batches)
	pred_lat = np.concatenate(pred_lat_batches)
	pred_lon = np.concatenate(pred_lon_batches)
	correct = np.concatenate(correct_batches)

	distances_km = haversine(true_lat, true_lon, pred_lat, pred_lon)
	return distances_km, correct


def print_report(distances_km: np.ndarray, correct: np.ndarray) -> None:
	print(f"Evaluated {len(distances_km):,} images")
	print(f"Top-1 geocell accuracy: {float(np.mean(correct)):.1%}")
	print(f"Distance error: mean={np.mean(distances_km):.1f} km, median={np.median(distances_km):.1f} km")
	print("Accuracy within distance thresholds:")
	for threshold_km, accuracy in accuracy_at_thresholds(distances_km).items():
		print(f"  <= {threshold_km:>7,.0f} km: {accuracy:.1%}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
	parser.add_argument("--dev-tfrecord", type=Path, default=DEV_TFRECORD)
	parser.add_argument("--batch-size", type=int, default=32)
	parser.add_argument("--normalization", choices=("minus_one_one", "imagenet"), default="minus_one_one")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	if not args.model_path.exists():
		raise FileNotFoundError(f"{args.model_path} not found; run train_2.py first")
	if not args.dev_tfrecord.exists():
		raise FileNotFoundError(
			f"{args.dev_tfrecord} not found; run data_pipeline_1.py with --dev-fraction to create it"
		)
	if not CENTROIDS_PATH.exists():
		raise FileNotFoundError(f"{CENTROIDS_PATH} not found; run data_pipeline_1.py first")

	model = tf.keras.models.load_model(args.model_path)
	centroids = load_centroids(CENTROIDS_PATH)
	dataset = build_eval_dataset(
		args.dev_tfrecord, batch_size=args.batch_size, normalization=args.normalization
	)

	distances_km, correct = evaluate(model, dataset, centroids)
	print_report(distances_km, correct)


if __name__ == "__main__":
	main()
