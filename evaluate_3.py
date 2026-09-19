"""Evaluate a trained geocell classifier using haversine distance error."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from data_pipeline_1 import CENTROIDS_PATH, build_eval_dataset
from geo_utils import accuracy_at_thresholds, haversine, load_centroids, reverse_geocode, weighted_centroid

DEV_TFRECORD = Path("data/processed/dev.tfrecord")
DEFAULT_MODEL_PATH = Path("checkpoints/best.keras")
COUNTRY_ACCURACY_PATH = Path("country_accuracy.json")


def build_cell_country_map(centroids: dict[int, tuple[float, float]]) -> dict[int, str]:
	"""Reverse-geocode each geocell centroid once to get its country code.

	One network call per geocell (not per dev-set image), so this is cheap even
	for a large dev set - typically dozens to a couple hundred calls total.
	"""
	print(f"Reverse-geocoding {len(centroids)} geocell centroids to countries...")
	return {cell_id: reverse_geocode(lat, lon)["country_code"] for cell_id, (lat, lon) in centroids.items()}


def evaluate(
	model: tf.keras.Model,
	dataset: tf.data.Dataset,
	centroids: dict[int, tuple[float, float]],
	cell_country_map: dict[int, str],
	top_k: int = 5,
) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, int]]]:
	"""Return (haversine_distances_km, top1_cell_correct, per_country_stats).

	Distance error uses a top-k probability-weighted centroid rather than a hard
	argmax, which smooths over near-miss cells; top-1 accuracy still reflects the
	single most-likely cell for a classification-style read on the model.
	per_country_stats maps each true country code seen in the dev set to
	{"correct": n, "total": n}, where "correct" means the predicted cell's own
	country matched the example's true country.
	"""
	num_classes = len(centroids)
	centroid_lat = np.array([centroids[i][0] for i in range(num_classes)])
	centroid_lon = np.array([centroids[i][1] for i in range(num_classes)])

	true_lat_batches, true_lon_batches = [], []
	pred_lat_batches, pred_lon_batches = [], []
	correct_batches = []
	country_correct: dict[str, int] = {}
	country_total: dict[str, int] = {}

	for images, cell_ids, latitudes, longitudes, country_codes in dataset:
		probabilities = model.predict(images, verbose=0)
		predicted_cells = np.argmax(probabilities, axis=1)
		pred_lat, pred_lon = weighted_centroid(probabilities, centroid_lat, centroid_lon, top_k=top_k)
		true_lat_batches.append(latitudes.numpy())
		true_lon_batches.append(longitudes.numpy())
		pred_lat_batches.append(pred_lat)
		pred_lon_batches.append(pred_lon)
		correct_batches.append(predicted_cells == cell_ids.numpy())

		for true_country_bytes, predicted_cell in zip(country_codes.numpy(), predicted_cells):
			true_country = true_country_bytes.decode("utf-8")
			country_total[true_country] = country_total.get(true_country, 0) + 1
			if cell_country_map.get(int(predicted_cell)) == true_country:
				country_correct[true_country] = country_correct.get(true_country, 0) + 1

	true_lat = np.concatenate(true_lat_batches)
	true_lon = np.concatenate(true_lon_batches)
	pred_lat = np.concatenate(pred_lat_batches)
	pred_lon = np.concatenate(pred_lon_batches)
	correct = np.concatenate(correct_batches)

	distances_km = haversine(true_lat, true_lon, pred_lat, pred_lon)
	country_stats = {
		country: {"correct": country_correct.get(country, 0), "total": total}
		for country, total in country_total.items()
	}
	return distances_km, correct, country_stats


def print_report(distances_km: np.ndarray, correct: np.ndarray, country_stats: dict[str, dict[str, int]]) -> None:
	print(f"Evaluated {len(distances_km):,} images")
	print(f"Top-1 geocell accuracy: {float(np.mean(correct)):.1%}")
	print(f"Distance error: mean={np.mean(distances_km):.1f} km, median={np.median(distances_km):.1f} km")
	print("Accuracy within distance thresholds:")
	for threshold_km, accuracy in accuracy_at_thresholds(distances_km).items():
		print(f"  <= {threshold_km:>7,.0f} km: {accuracy:.1%}")
	print(f"Per-country accuracy ({len(country_stats)} countries with dev data):")
	for country, stats in sorted(country_stats.items(), key=lambda item: -item[1]["correct"] / item[1]["total"]):
		accuracy = stats["correct"] / stats["total"]
		print(f"  {country}: {accuracy:.1%} ({stats['correct']}/{stats['total']})")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
	parser.add_argument("--dev-tfrecord", type=Path, default=DEV_TFRECORD)
	parser.add_argument("--batch-size", type=int, default=32)
	parser.add_argument("--normalization", choices=("minus_one_one", "imagenet"), default="minus_one_one")
	parser.add_argument("--top-k", type=int, default=5, help="Cells averaged for the weighted-centroid guess")
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
	cell_country_map = build_cell_country_map(centroids)
	dataset = build_eval_dataset(
		args.dev_tfrecord, batch_size=args.batch_size, normalization=args.normalization
	)

	distances_km, correct, country_stats = evaluate(
		model, dataset, centroids, cell_country_map, top_k=args.top_k
	)
	print_report(distances_km, correct, country_stats)

	COUNTRY_ACCURACY_PATH.write_text(json.dumps(country_stats, indent=2) + "\n", encoding="utf-8")
	print(f"\nWrote per-country accuracy to {COUNTRY_ACCURACY_PATH}")


if __name__ == "__main__":
	main()
