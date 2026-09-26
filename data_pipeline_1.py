"""Stream, materialize, and prepare the Asia-wide street-view dataset."""

# =============================================================================
# data_pipeline_1.py - STEP 1 of 5: get the data and prepare it for training
# -----------------------------------------------------------------------------
# Run order for the whole project:
#   1. data_pipeline_1.py  <- this file
#   2. train_2.py          (train the model)
#   3. evaluate_3.py       (measure how accurate it is)
#   4. export_4.py         (shrink it for the web app)
#   5. streamlit run app.py
#
# What this file does, in order (see main() at the bottom):
#   1. STREAM   the Hugging Face dataset record by record, keeping only photos
#               from the countries listed in COUNTRIES, and cache them on disk.
#   2. LOAD     that cache back into memory.
#   3. BASELINE print how bad a "dumb" model would be, as a comparison point.
#   4. CLUSTER  every photo's GPS location into NUM_GEOCELLS groups ("geocells")
#               using K-means. This turns "guess exact coordinates" (very hard)
#               into "guess which of 100 regions" (a normal classification task).
#   5. LABEL    every photo with its geocell number and save it again.
#   6. SPLIT    into training and development ("dev") sets, if --dev-fraction
#               is given, keeping all views of one location on the same side.
#
# It also defines build_tf_dataset() and build_eval_dataset(), which train_2.py
# and evaluate_3.py import to read the saved files back efficiently.
#
# IMPORTANT: run this from a terminal with the flag, e.g.
#     python data_pipeline_1.py --dev-fraction 0.2
# (VS Code's "Run" button does not pass flags, so no train/dev split is made.)
# =============================================================================

from __future__ import annotations

import argparse                        # read command-line flags like --dev-fraction
import hashlib                         # SHA-1 hashing, used for a repeatable train/dev split
import io                              # in-memory byte buffer for re-encoding images
import json                            # write cell_centroids.json
from collections import Counter        # count images per country
from pathlib import Path               # file paths
from typing import Any, Iterator       # type hints

import numpy as np                     # numeric arrays
import tensorflow as tf                # TFRecord files and image decoding
from datasets import load_dataset      # Hugging Face's dataset downloader/streamer
from sklearn.cluster import KMeans     # the clustering algorithm that creates geocells

from geo_utils import haversine        # distance in km between two GPS points


# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------

# The dataset on Hugging Face: street-view photos from around the world, each
# labelled with the latitude, longitude and country it was taken in.
DATASET_NAME = "josefbednar/world-streetview-500k"
# ISO 3166-1 alpha-2 codes for every country/territory in the UN geoscheme's Asia
# region (Eastern, South-Eastern, Southern, Central, and Western Asia). Set this to
# None instead to remove the filter entirely and keep every country in the dataset.
# (A frozenset is an unchangeable set; checking "is X in it?" is very fast.)
COUNTRIES: frozenset[str] | None = frozenset(
	{
		# Eastern Asia
		"CN", "HK", "MO", "JP", "MN", "KP", "KR", "TW",
		# South-Eastern Asia
		"BN", "KH", "TL", "ID", "LA", "MY", "MM", "PH", "SG", "TH", "VN",
		# Southern Asia
		"AF", "BD", "BT", "IN", "IR", "MV", "NP", "PK", "LK",
		# Central Asia
		"KZ", "KG", "TJ", "TM", "UZ",
		# Western Asia
		"AM", "AZ", "BH", "CY", "GE", "IQ", "IL", "JO", "KW", "LB",
		"OM", "PS", "QA", "SA", "SY", "TR", "AE", "YE",
	}
)
# How many geographic groups (geocells) K-means should create by default.
# This is also the number of classes the model learns to choose between.
# Can be overridden with --num-geocells.
NUM_GEOCELLS = 100
# Every photo is resized to 224 x 224 pixels, the input size MobileNetV2 expects.
IMAGE_SIZE = (224, 224)
# Where the streamed, filtered photos are cached (before geocells are added).
RAW_TFRECORD = Path("data/raw/filtered_train.tfrecord")
# The same photos after each one has been labelled with its geocell number.
PROCESSED_TFRECORD = Path("data/processed/filtered_train_with_cells.tfrecord")
# The centre point (latitude, longitude) of every geocell, saved as JSON.
CENTROIDS_PATH = Path("cell_centroids.json")


# -----------------------------------------------------------------------------
# TFRecord helpers
# -----------------------------------------------------------------------------
# A TFRecord file is TensorFlow's own binary format for storing large datasets:
# one long file holding many "Example" records back to back. It is much faster
# to read during training than thousands of separate image files.
#
# Each Example is a set of named "features", and each feature must be one of
# three types: bytes, floats, or integers. The four small helpers below wrap a
# normal Python value in the right feature type.

def _bytes_feature(value: bytes) -> tf.train.Feature:
	# For raw bytes, such as the JPEG image data.
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value: float) -> tf.train.Feature:
	# For decimal numbers, such as latitude and longitude.
	return tf.train.Feature(float_list=tf.train.FloatList(value=[float(value)]))


def _int_feature(value: int) -> tf.train.Feature:
	# For whole numbers, such as the geocell id.
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))


def _string_feature(value: str) -> tf.train.Feature:
	# TFRecord has no text type, so text is stored as UTF-8 encoded bytes.
	return _bytes_feature(value.encode("utf-8"))


def _image_to_jpeg(image: Any) -> bytes:
	"""Convert a datasets Image value to compact RGB JPEG bytes."""
	# Hugging Face may hand us the image in different forms depending on how the
	# dataset was stored. This function turns any of them into the same thing:
	# JPEG bytes in RGB colour (3 channels), so every stored image is consistent.

	# Form 1: a dictionary like {"bytes": b"...", "path": ...} -> take the bytes.
	if isinstance(image, dict):
		image = image.get("bytes")
	# Form 2: raw encoded bytes -> decode with TensorFlow (forcing 3 colour
	# channels, which drops any transparency) and re-encode as JPEG.
	if isinstance(image, bytes):
		image = tf.io.decode_image(image, channels=3, expand_animations=False)
		return bytes(tf.io.encode_jpeg(image).numpy())
	# Form 3: a PIL (Python Imaging Library) image object -> convert to RGB and
	# save as JPEG into an in-memory buffer instead of a file on disk.
	if hasattr(image, "convert"):
		output = io.BytesIO()
		image.convert("RGB").save(output, format="JPEG", quality=95)
		return output.getvalue()
	# Anything else is unexpected, so fail loudly rather than save bad data.
	raise TypeError(f"Unsupported image value: {type(image)!r}")


def _example_from_record(record: dict[str, Any], cell_id: int | None = None) -> tf.train.Example:
	# Package one photo's data (a Python dictionary) into a TFRecord Example,
	# ready to be written to disk.
	features = {
		"image": _bytes_feature(record["image"]),                   # the JPEG itself
		"image_id": _string_feature(str(record["image_id"])),       # unique photo id
		"panoid": _string_feature(str(record["panoid"])),           # panorama id (see split_by_panoid)
		"country_code": _string_feature(str(record["country_code"])),  # e.g. "VN"
		"latitude": _float_feature(record["latitude"]),             # true location...
		"longitude": _float_feature(record["longitude"]),           # ...of the photo
		"elevation": _float_feature(record.get("elevation", 0.0)),  # height above sea level
	}
	# The geocell label only exists after clustering (step 4), so it is optional:
	# the raw cache is written without it, the processed file is written with it.
	if cell_id is not None:
		features["cell_id"] = _int_feature(cell_id)
	return tf.train.Example(features=tf.train.Features(feature=features))


def _parse_example(serialized: bytes) -> dict[str, Any]:
	# The reverse of _example_from_record: turn one stored Example back into a
	# plain Python dictionary, so the rest of this file can work with it easily.
	example = tf.train.Example.FromString(serialized)
	features = example.features.feature

	def text(name: str) -> str:
		# Text was stored as bytes, so decode it back into a string.
		return features[name].bytes_list.value[0].decode("utf-8")

	# Each feature holds a list; [0] takes its single value.
	result: dict[str, Any] = {
		"image": bytes(features["image"].bytes_list.value[0]),
		"image_id": text("image_id"),
		"panoid": text("panoid"),
		"country_code": text("country_code"),
		"latitude": features["latitude"].float_list.value[0],
		"longitude": features["longitude"].float_list.value[0],
		"elevation": features["elevation"].float_list.value[0],
	}
	if "cell_id" in features:
		result["cell_id"] = features["cell_id"].int64_list.value[0]
	return result


def _iter_records(path: Path) -> Iterator[dict[str, Any]]:
	# Read a TFRecord file one Example at a time and yield each as a dictionary.
	# `yield` makes this a generator: records are produced one by one as they are
	# needed, rather than all being built up front.
	dataset = tf.data.TFRecordDataset(str(path), num_parallel_reads=1)
	for serialized in dataset.as_numpy_iterator():
		yield _parse_example(serialized)


# -----------------------------------------------------------------------------
# Step 1: stream the dataset and cache the photos we want
# -----------------------------------------------------------------------------

def materialize_filtered_split(output_path: Path = RAW_TFRECORD) -> None:
	"""Stream the source once and cache the filtered records as TFRecord."""
	# Downloading the entire dataset first would take tens of gigabytes of disk.
	# Instead, "streaming" pulls records from Hugging Face one at a time. We check
	# each record's country, keep the ones we want, and write them straight into
	# a local TFRecord cache. That way this slow download only ever happens once.

	# If the cache already exists, skip the download entirely.
	# NOTE: this only checks that the file exists - not which countries it holds.
	# After changing COUNTRIES, delete data/raw/filtered_train.tfrecord first,
	# otherwise the old cache is silently reused.
	if output_path.exists() and output_path.stat().st_size > 0:
		print(f"Using existing local cache: {output_path}")
		return

	# Create data/raw/ if it doesn't exist yet.
	output_path.parent.mkdir(parents=True, exist_ok=True)
	counts: Counter[str] = Counter()   # images kept per country, e.g. {"JP": 5896, ...}
	total = 0                          # images kept so far
	scanned = 0                        # records looked at so far (kept or not)
	progress_every = 5_000             # print a progress line this often
	# streaming=True: don't download everything first - iterate over it lazily.
	streamed = load_dataset(DATASET_NAME, split="train", streaming=True)
	# `with` makes sure the file is properly closed and saved at the end,
	# even if an error happens part-way through.
	with tf.io.TFRecordWriter(str(output_path)) as writer:
		for source_record in streamed:
			scanned += 1
			# This loop can run for many hours, so print regular progress so it
			# doesn't look frozen.
			if scanned % progress_every == 0:
				print(f"  scanned {scanned:,} source records, kept {total:,} so far...")
			# THE FILTER: skip any photo whose country isn't in COUNTRIES.
			# (Note: every record still has to be downloaded to read its country,
			# so filtering reduces disk space used, not download time.)
			if COUNTRIES is not None and source_record.get("country_code") not in COUNTRIES:
				continue
			# Build a clean record with just the fields we need, with the image
			# normalised to JPEG and the numbers forced to floats.
			record = {
				"image": _image_to_jpeg(source_record["image"]),
				"image_id": source_record["image_id"],
				"panoid": source_record["panoid"],
				"country_code": source_record["country_code"],
				"latitude": float(source_record["latitude"]),
				"longitude": float(source_record["longitude"]),
				# Some records have no elevation (None) - store 0.0 instead.
				"elevation": float(source_record.get("elevation") or 0.0),
			}
			# Convert to a TFRecord Example and append it to the cache file.
			writer.write(_example_from_record(record).SerializeToString())
			counts[record["country_code"]] += 1
			total += 1

	# Summary once the whole dataset has been streamed.
	print(f"Scanned {scanned:,} source records total")
	print(f"Materialized {total:,} filtered images to {output_path}")
	print_country_counts(counts)


def print_country_counts(counts: Counter[str]) -> None:
	# Print how many images each country has. Countries with very few images
	# get a warning: the model will struggle to learn places it has barely seen.
	print("Per-country image counts:")
	# With no COUNTRIES filter, list whatever countries actually showed up rather than
	# a fixed set - there's no predetermined list to iterate over.
	countries_to_show = sorted(COUNTRIES) if COUNTRIES is not None else sorted(counts)
	for country in countries_to_show:
		count = counts[country]
		warning = "  WARNING: under ~2,000 images" if count < 2_000 else ""
		print(f"  {country}: {count:,}{warning}")


# -----------------------------------------------------------------------------
# Step 2: load the cache back into memory
# -----------------------------------------------------------------------------

def load_cached_records(path: Path = RAW_TFRECORD) -> list[dict[str, Any]]:
	# Read every cached photo into one Python list.
	# NOTE: this loads ALL of them into RAM at once. That works for the Asia
	# subset, but it is exactly what ran out of memory when the whole world was
	# tried - that would need a streaming approach instead.
	records = list(_iter_records(path))
	print_country_counts(Counter(record["country_code"] for record in records))
	if not records:
		raise RuntimeError(f"No filtered records found in {path}")
	return records


# -----------------------------------------------------------------------------
# Step 4: create the geocells with K-means clustering
# -----------------------------------------------------------------------------

def cluster_geocells(
	records: list[dict[str, Any]],
	centroids_path: Path = CENTROIDS_PATH,
	n_clusters: int = NUM_GEOCELLS,
) -> dict[int, tuple[float, float]]:
	# WHY: predicting exact coordinates directly is a very hard problem. Instead
	# we split the map into n_clusters regions ("geocells") and train the model
	# to answer the easier question "which region is this photo from?".
	#
	# HOW K-MEANS WORKS:
	#   1. Place n_clusters starting centre points (smart random guesses).
	#   2. Assign every photo's GPS point to its nearest centre.
	#   3. Move each centre to the average position of the points assigned to it.
	#   4. Repeat steps 2-3 until the centres stop moving.
	# The final centres are the "centroids". Areas with lots of photos end up with
	# many small geocells; sparse areas get fewer, larger ones.

	# K-means can't make more groups than there are points.
	if len(records) < n_clusters:
		raise ValueError(f"Need at least {n_clusters} images for KMeans, found {len(records)}")
	# Build a table with one row per photo and two columns: [latitude, longitude].
	coordinates = np.array(
		[[record["latitude"], record["longitude"]] for record in records], dtype=np.float64
	)
	# random_state=42 fixes the random starting points, so running this again on
	# the same data gives exactly the same geocells (reproducible results).
	# n_init=10 runs K-means 10 times from different starts and keeps the best,
	# because an unlucky starting guess can produce poor clusters.
	model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
	# fit_predict finds the clusters AND returns which cluster each photo is in.
	labels = model.fit_predict(coordinates)
	# Attach that geocell number to each photo - this becomes the training label.
	for record, label in zip(records, labels):
		record["cell_id"] = int(label)

	# Save every geocell's centre point to JSON, e.g.
	#   {"0": {"centroid_lat": 21.03, "centroid_lon": 105.85}, ...}
	# This file is the "dictionary" that later turns a predicted geocell number
	# back into a real location on the map.
	centroids = {
		str(cell_id): {
			"centroid_lat": float(centroid[0]),
			"centroid_lon": float(centroid[1]),
		}
		for cell_id, centroid in enumerate(model.cluster_centers_)
	}
	centroids_path.write_text(json.dumps(centroids, indent=2) + "\n", encoding="utf-8")
	print(f"Saved {n_clusters} geocell centroids to {centroids_path}")
	return {int(key): (value["centroid_lat"], value["centroid_lon"]) for key, value in centroids.items()}


# -----------------------------------------------------------------------------
# Step 5: save the labelled photos
# -----------------------------------------------------------------------------

def write_annotated_records(records: list[dict[str, Any]], output_path: Path = PROCESSED_TFRECORD) -> None:
	# Write photos back out to a TFRecord file, this time including each photo's
	# geocell label (cell_id), so training can read image + label together.
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with tf.io.TFRecordWriter(str(output_path)) as writer:
		for record in records:
			writer.write(_example_from_record(record, record["cell_id"]).SerializeToString())
	print(f"Wrote annotated records to {output_path}")


# -----------------------------------------------------------------------------
# Step 6: split into training and development (dev) sets
# -----------------------------------------------------------------------------

def split_by_panoid(
	records: list[dict[str, Any]], dev_fraction: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
	"""Split complete panoids, so views from one location never cross the split."""
	# The model trains on the "train" set and is scored on the "dev" set, which it
	# never sees during training. That tells us how well it handles NEW photos.
	#
	# DATA LEAKAGE: several photos can come from the same panorama (the same spot,
	# facing different directions). If one view of a spot were in training and
	# another view of the same spot were in dev, the model could "recognise" the
	# place instead of genuinely generalising, and the dev score would look better
	# than it really is. So the split is done by panorama id (panoid): every view
	# of a location goes to the same side.
	if not 0.0 <= dev_fraction < 1.0:
		raise ValueError("dev_fraction must be in [0, 1)")
	train_records, dev_records = [], []
	for record in records:
		# Hash the panorama id with SHA-1. A hash always gives the same output for
		# the same input, but the outputs look random - so this is a repeatable
		# "coin flip" per panorama: same split every run, no randomness to manage.
		digest = hashlib.sha1(str(record["panoid"]).encode("utf-8")).digest()
		# Turn the first 8 bytes of the hash into a number between 0 and 1.
		# If it's below dev_fraction (e.g. 0.2), this panorama goes to dev;
		# on average that sends 20% of panoramas to dev and 80% to train.
		goes_to_dev = int.from_bytes(digest[:8], "big") / 2**64 < dev_fraction
		(dev_records if goes_to_dev else train_records).append(record)
	return train_records, dev_records


def write_split_records(records: list[dict[str, Any]], directory: Path, dev_fraction: float) -> None:
	# With --dev-fraction 0 (the default) no split is made, and training falls
	# back to using every photo with no validation set.
	if dev_fraction == 0.0:
		return
	train_records, dev_records = split_by_panoid(records, dev_fraction)
	write_annotated_records(train_records, directory / "train.tfrecord")
	write_annotated_records(dev_records, directory / "dev.tfrecord")
	print(f"Internal panoid split: {len(train_records):,} train / {len(dev_records):,} dev images")


# -----------------------------------------------------------------------------
# Reading the data back during training and evaluation
# -----------------------------------------------------------------------------
# The functions below are not called by this file's main(); train_2.py and
# evaluate_3.py import them. They build a tf.data pipeline: a conveyor belt that
# reads, decodes, resizes and batches images on the fly while the model trains,
# so the full dataset never has to sit in memory as decoded pixels.

def _preprocess_image(image_bytes: tf.Tensor, normalization: str, training: bool) -> tf.Tensor:
	# Turn stored JPEG bytes into the number grid the neural network needs.

	# Decode JPEG -> a 3-D grid of pixels (height x width x 3 colour channels).
	image = tf.io.decode_jpeg(image_bytes, channels=3)
	# Convert pixel values from whole numbers 0-255 to decimals 0.0-1.0.
	image = tf.image.convert_image_dtype(image, tf.float32)
	if training:
		# DATA AUGMENTATION (training only): show the model a slightly different
		# version of each photo every epoch, so it learns general features rather
		# than memorising exact pixels.
		#   - resize a little larger (232x232), then cut a random 224x224 piece
		#     out of it -> the framing shifts slightly each time
		#   - randomly mirror the photo left-to-right
		image = tf.image.resize(image, [232, 232])
		image = tf.image.random_crop(image, [IMAGE_SIZE[0], IMAGE_SIZE[1], 3])
		image = tf.image.random_flip_left_right(image)
	else:
		# Evaluation must be repeatable, so just resize - no randomness.
		image = tf.image.resize(image, IMAGE_SIZE)
	# NORMALIZATION: rescale pixel values to the range the pretrained model was
	# originally trained with.
	if normalization == "minus_one_one":
		# 0..1 -> -1..1 (what MobileNetV2 expects).
		image = image * 2.0 - 1.0
	else:
		# "imagenet": subtract the average colour of the ImageNet dataset and
		# divide by its spread (standard deviation), per colour channel.
		image = (image - tf.constant([0.485, 0.456, 0.406])) / tf.constant(
			[0.229, 0.224, 0.225]
		)
	return image


def build_tf_dataset(
	tfrecord_path: str | Path,
	batch_size: int = 32,
	normalization: str = "minus_one_one",
	training: bool = True,
	shuffle: bool = True,
) -> tf.data.Dataset:
	"""Build a decoded 224x224 image pipeline returning images and cell labels."""
	# Used by train_2.py. Produces (image, geocell label) pairs in batches.
	if normalization not in {"minus_one_one", "imagenet"}:
		raise ValueError("normalization must be 'minus_one_one' or 'imagenet'")
	# Tell TensorFlow which stored fields to read and what type each one is.
	# Only the image and the label are needed for training.
	feature_spec = {
		"image": tf.io.FixedLenFeature([], tf.string),
		"cell_id": tf.io.FixedLenFeature([], tf.int64),
	}

	def parse(serialized: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
		# Applied to every record: unpack it, prepare the image, return the pair.
		parsed = tf.io.parse_single_example(serialized, feature_spec)
		image = _preprocess_image(parsed["image"], normalization, training)
		return image, tf.cast(parsed["cell_id"], tf.int32)

	dataset = tf.data.TFRecordDataset(str(tfrecord_path))
	if shuffle:
		# Mix up the order using a buffer of 2,048 records, reshuffled every
		# epoch, so the model doesn't see photos in the same order each time.
		dataset = dataset.shuffle(2_048, reshuffle_each_iteration=True)
	# map     -> run parse() on every record, using several CPU cores in parallel
	# batch   -> group records into batches (e.g. 32 photos at a time)
	# prefetch-> prepare the next batch while the model is busy with this one
	return dataset.map(parse, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(
		tf.data.AUTOTUNE
	)


def build_eval_dataset(
	tfrecord_path: str | Path,
	batch_size: int = 32,
	normalization: str = "minus_one_one",
) -> tf.data.Dataset:
	"""Build an unaugmented pipeline returning images, cell labels, true coordinates, and country."""
	# Used by evaluate_3.py. Unlike build_tf_dataset it:
	#   - never augments or shuffles (scores must be repeatable)
	#   - also returns the TRUE latitude, longitude and country of each photo,
	#     which evaluation needs to measure distance error in km and
	#     accuracy per country.
	if normalization not in {"minus_one_one", "imagenet"}:
		raise ValueError("normalization must be 'minus_one_one' or 'imagenet'")
	feature_spec = {
		"image": tf.io.FixedLenFeature([], tf.string),
		"cell_id": tf.io.FixedLenFeature([], tf.int64),
		"latitude": tf.io.FixedLenFeature([], tf.float32),
		"longitude": tf.io.FixedLenFeature([], tf.float32),
		"country_code": tf.io.FixedLenFeature([], tf.string),
	}

	def parse(serialized: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
		parsed = tf.io.parse_single_example(serialized, feature_spec)
		image = _preprocess_image(parsed["image"], normalization, training=False)
		return (
			image,
			tf.cast(parsed["cell_id"], tf.int32),
			parsed["latitude"],
			parsed["longitude"],
			parsed["country_code"],
		)

	dataset = tf.data.TFRecordDataset(str(tfrecord_path))
	return dataset.map(parse, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(
		tf.data.AUTOTUNE
	)


# -----------------------------------------------------------------------------
# Step 3: a simple baseline to compare the model against
# -----------------------------------------------------------------------------

def print_baseline(records: list[dict[str, Any]]) -> None:
	# A "baseline" is the score of the dumbest reasonable strategy. Here: always
	# guess the single average location of all photos, no matter what the photo
	# shows. If the trained model can't beat this, it hasn't learned anything
	# useful from the images.
	latitudes = np.array([record["latitude"] for record in records], dtype=np.float64)
	longitudes = np.array([record["longitude"] for record in records], dtype=np.float64)
	# The one fixed guess: the average latitude and average longitude.
	centroid_lat, centroid_lon = float(latitudes.mean()), float(longitudes.mean())
	# How far that fixed guess is from every photo's real location.
	distances = haversine(centroid_lat, centroid_lon, latitudes, longitudes)
	print(
		"Global-centroid baseline: "
		f"mean={float(np.mean(distances)):.2f} km, "
		f"median={float(np.median(distances)):.2f} km"
	)


# -----------------------------------------------------------------------------
# Command-line interface
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
	# Read optional flags given when running the script, for example:
	#     python data_pipeline_1.py --dev-fraction 0.2 --num-geocells 30
	parser = argparse.ArgumentParser(description=__doc__)
	# Hidden safety option; only this one dataset is supported.
	parser.add_argument("--dataset", default=DATASET_NAME, help=argparse.SUPPRESS)
	# Fraction of panoramas to hold out for evaluation (0 = no split).
	parser.add_argument("--dev-fraction", type=float, default=0.0)
	# Only affects the printed hint at the end; training has its own flag.
	parser.add_argument("--normalization", choices=("minus_one_one", "imagenet"), default="minus_one_one")
	parser.add_argument("--batch-size", type=int, default=32)
	# How many geocells (classes) K-means should create.
	parser.add_argument("--num-geocells", type=int, default=NUM_GEOCELLS)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	if args.dataset != DATASET_NAME:
		raise ValueError("This pipeline is defined for the expected World Streetview dataset")
	materialize_filtered_split()                                   # 1. stream + cache
	records = load_cached_records()                                # 2. load into memory
	print_baseline(records)                                        # 3. baseline score
	cluster_geocells(records, n_clusters=args.num_geocells)        # 4. K-means geocells
	write_annotated_records(records)                               # 5. save with labels
	write_split_records(records, Path("data/processed"), args.dev_fraction)  # 6. train/dev split
	print(
		f"Ready: build_tf_dataset('{PROCESSED_TFRECORD}', "
		f"batch_size={args.batch_size}, normalization='{args.normalization}')"
	)


# Only run main() when this file is executed directly (python data_pipeline_1.py),
# not when another file imports functions from it (as train_2.py does).
if __name__ == "__main__":
	main()
