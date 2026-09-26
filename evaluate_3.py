"""Evaluate a trained geocell classifier using haversine distance error."""

# =============================================================================
# evaluate_3.py - STEP 3 of 5: measure how accurate the trained model is
# -----------------------------------------------------------------------------
# Input : checkpoints/best.keras        (from train_2.py)
#         data/processed/dev.tfrecord   (from data_pipeline_1.py --dev-fraction)
#         cell_centroids.json           (from data_pipeline_1.py)
# Output: printed report + country_accuracy.json (shown on the app's
#         "Model Stats" page, after export_4.py copies it into export/)
#
# The model is scored on the DEV set - photos it never saw during training -
# so the numbers reflect how it will perform on new, real-world photos.
#
# METRICS REPORTED
#   1. Top-1 geocell accuracy: % of photos where the single most-likely geocell
#      was exactly the right one. Strict - a guess one geocell away counts as wrong.
#   2. Distance error (km): how far the guessed coordinates were from the true
#      location, as a mean and a median. This is more forgiving and more
#      meaningful: being 50 km off is much better than 3,000 km off, even though
#      both count as "wrong" under top-1 accuracy.
#   3. Accuracy within distance thresholds: % of guesses within 1 / 25 / 200 /
#      750 / 2500 km (roughly street / city / region / country / continent).
#   4. Per-country accuracy: for photos truly from country X, how often the
#      predicted geocell was also in country X.
# =============================================================================

from __future__ import annotations

import argparse            # read command-line flags
import json                # write country_accuracy.json
from pathlib import Path   # file paths

import numpy as np         # numeric arrays
import tensorflow as tf    # load and run the trained model

# The evaluation data pipeline (no augmentation; includes true coordinates).
from data_pipeline_1 import CENTROIDS_PATH, build_eval_dataset
from geo_utils import accuracy_at_thresholds, haversine, load_centroids, reverse_geocode, weighted_centroid

DEV_TFRECORD = Path("data/processed/dev.tfrecord")
# best.keras = the best-scoring version saved during training.
DEFAULT_MODEL_PATH = Path("checkpoints/best.keras")
COUNTRY_ACCURACY_PATH = Path("country_accuracy.json")


def build_cell_country_map(centroids: dict[int, tuple[float, float]]) -> dict[int, str]:
	"""Reverse-geocode each geocell centroid once to get its country code.

	One network call per geocell (not per dev-set image), so this is cheap even
	for a large dev set - typically dozens to a couple hundred calls total.
	"""
	# To score "did the model get the country right?", we need to know which
	# country each geocell belongs to. Geocells are only coordinates, so we look
	# up each geocell's centre point with the reverse-geocoding web service.
	# Result: {0: "VN", 1: "JP", 2: "PH", ...}
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
	# Put every geocell's centre into two arrays, where position i = geocell i.
	# This lets weighted_centroid() look up many locations at once.
	num_classes = len(centroids)
	centroid_lat = np.array([centroids[i][0] for i in range(num_classes)])
	centroid_lon = np.array([centroids[i][1] for i in range(num_classes)])

	# Results are collected batch by batch in these lists, then joined together
	# at the end.
	true_lat_batches, true_lon_batches = [], []
	pred_lat_batches, pred_lon_batches = [], []
	correct_batches = []
	# Running tallies per true country: how many photos, and how many correct.
	country_correct: dict[str, int] = {}
	country_total: dict[str, int] = {}

	# Loop over the dev set one batch (e.g. 32 photos) at a time.
	for images, cell_ids, latitudes, longitudes, country_codes in dataset:
		# Run the model: one row of probabilities (one per geocell) for each photo.
		probabilities = model.predict(images, verbose=0)
		# argmax = the index of the highest probability = the top-1 geocell.
		predicted_cells = np.argmax(probabilities, axis=1)
		# Turn the probabilities into one guessed lat/lon per photo, by blending
		# the top-k geocells (see geo_utils.weighted_centroid for the details).
		pred_lat, pred_lon = weighted_centroid(probabilities, centroid_lat, centroid_lon, top_k=top_k)
		# .numpy() converts TensorFlow tensors into ordinary numpy arrays.
		true_lat_batches.append(latitudes.numpy())
		true_lon_batches.append(longitudes.numpy())
		pred_lat_batches.append(pred_lat)
		pred_lon_batches.append(pred_lon)
		# True/False per photo: was the top-1 geocell exactly the correct one?
		correct_batches.append(predicted_cells == cell_ids.numpy())

		# Per-country scoring, one photo at a time.
		for true_country_bytes, predicted_cell in zip(country_codes.numpy(), predicted_cells):
			# The country code is stored as bytes (b"VN"); decode it to text ("VN").
			true_country = true_country_bytes.decode("utf-8")
			country_total[true_country] = country_total.get(true_country, 0) + 1
			# Correct if the predicted geocell sits in the photo's real country.
			if cell_country_map.get(int(predicted_cell)) == true_country:
				country_correct[true_country] = country_correct.get(true_country, 0) + 1

	# Join the per-batch pieces into single long arrays covering the whole dev set.
	true_lat = np.concatenate(true_lat_batches)
	true_lon = np.concatenate(true_lon_batches)
	pred_lat = np.concatenate(pred_lat_batches)
	pred_lon = np.concatenate(pred_lon_batches)
	correct = np.concatenate(correct_batches)

	# Distance in km between every guess and its true location, all at once.
	distances_km = haversine(true_lat, true_lon, pred_lat, pred_lon)
	# Combine the two tallies into {"VN": {"correct": 423, "total": 442}, ...}
	country_stats = {
		country: {"correct": country_correct.get(country, 0), "total": total}
		for country, total in country_total.items()
	}
	return distances_km, correct, country_stats


def print_report(distances_km: np.ndarray, correct: np.ndarray, country_stats: dict[str, dict[str, int]]) -> None:
	# Print all the metrics in a readable format.
	print(f"Evaluated {len(distances_km):,} images")
	# The mean of True/False values = the fraction that were True.
	print(f"Top-1 geocell accuracy: {float(np.mean(correct)):.1%}")
	# Median is often more informative than mean here: a few wildly wrong guesses
	# (thousands of km off) pull the mean up a lot, but barely move the median.
	print(f"Distance error: mean={np.mean(distances_km):.1f} km, median={np.median(distances_km):.1f} km")
	print("Accuracy within distance thresholds:")
	for threshold_km, accuracy in accuracy_at_thresholds(distances_km).items():
		print(f"  <= {threshold_km:>7,.0f} km: {accuracy:.1%}")
	print(f"Per-country accuracy ({len(country_stats)} countries with dev data):")
	# Sort from most to least accurate (the minus sign flips the sort order).
	for country, stats in sorted(country_stats.items(), key=lambda item: -item[1]["correct"] / item[1]["total"]):
		accuracy = stats["correct"] / stats["total"]
		print(f"  {country}: {accuracy:.1%} ({stats['correct']}/{stats['total']})")


def parse_args() -> argparse.Namespace:
	# Optional command-line flags, e.g. python evaluate_3.py --top-k 3
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
	parser.add_argument("--dev-tfrecord", type=Path, default=DEV_TFRECORD)
	parser.add_argument("--batch-size", type=int, default=32)
	# Must match the normalization used during training.
	parser.add_argument("--normalization", choices=("minus_one_one", "imagenet"), default="minus_one_one")
	parser.add_argument("--top-k", type=int, default=5, help="Cells averaged for the weighted-centroid guess")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	# Check every required input exists first, with a clear message saying which
	# earlier step to run if one is missing.
	if not args.model_path.exists():
		raise FileNotFoundError(f"{args.model_path} not found; run train_2.py first")
	if not args.dev_tfrecord.exists():
		raise FileNotFoundError(
			f"{args.dev_tfrecord} not found; run data_pipeline_1.py with --dev-fraction to create it"
		)
	if not CENTROIDS_PATH.exists():
		raise FileNotFoundError(f"{CENTROIDS_PATH} not found; run data_pipeline_1.py first")

	# Load the trained model, the geocell centres and their countries, and the
	# dev-set pipeline.
	model = tf.keras.models.load_model(args.model_path)
	centroids = load_centroids(CENTROIDS_PATH)
	cell_country_map = build_cell_country_map(centroids)
	dataset = build_eval_dataset(
		args.dev_tfrecord, batch_size=args.batch_size, normalization=args.normalization
	)

	# Score the model and print the results.
	distances_km, correct, country_stats = evaluate(
		model, dataset, centroids, cell_country_map, top_k=args.top_k
	)
	print_report(distances_km, correct, country_stats)

	# Save the per-country numbers. export_4.py copies this file into export/,
	# where the Streamlit app's Model Stats page reads it.
	COUNTRY_ACCURACY_PATH.write_text(json.dumps(country_stats, indent=2) + "\n", encoding="utf-8")
	print(f"\nWrote per-country accuracy to {COUNTRY_ACCURACY_PATH}")


# Run main() only when this file is executed directly (python evaluate_3.py).
if __name__ == "__main__":
	main()
