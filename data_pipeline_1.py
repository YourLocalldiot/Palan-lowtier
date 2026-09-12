"""Stream, materialize, and prepare the four-country street-view dataset."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import tensorflow as tf
from datasets import load_dataset
from sklearn.cluster import KMeans

from geo_utils import haversine


DATASET_NAME = "josefbednar/world-streetview-500k"
COUNTRIES = frozenset({"VN", "PH", "JP", "KR"})
NUM_GEOCELLS = 100
IMAGE_SIZE = (224, 224)
RAW_TFRECORD = Path("data/raw/filtered_train.tfrecord")
PROCESSED_TFRECORD = Path("data/processed/filtered_train_with_cells.tfrecord")
CENTROIDS_PATH = Path("cell_centroids.json")


def _bytes_feature(value: bytes) -> tf.train.Feature:
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value: float) -> tf.train.Feature:
	return tf.train.Feature(float_list=tf.train.FloatList(value=[float(value)]))


def _int_feature(value: int) -> tf.train.Feature:
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))


def _string_feature(value: str) -> tf.train.Feature:
	return _bytes_feature(value.encode("utf-8"))


def _image_to_jpeg(image: Any) -> bytes:
	"""Convert a datasets Image value to compact RGB JPEG bytes."""
	if isinstance(image, dict):
		image = image.get("bytes")
	if isinstance(image, bytes):
		image = tf.io.decode_image(image, channels=3, expand_animations=False)
		return bytes(tf.io.encode_jpeg(image).numpy())
	if hasattr(image, "convert"):
		output = io.BytesIO()
		image.convert("RGB").save(output, format="JPEG", quality=95)
		return output.getvalue()
	raise TypeError(f"Unsupported image value: {type(image)!r}")


def _example_from_record(record: dict[str, Any], cell_id: int | None = None) -> tf.train.Example:
	features = {
		"image": _bytes_feature(record["image"]),
		"image_id": _string_feature(str(record["image_id"])),
		"panoid": _string_feature(str(record["panoid"])),
		"country_code": _string_feature(str(record["country_code"])),
		"latitude": _float_feature(record["latitude"]),
		"longitude": _float_feature(record["longitude"]),
		"elevation": _float_feature(record.get("elevation", 0.0)),
	}
	if cell_id is not None:
		features["cell_id"] = _int_feature(cell_id)
	return tf.train.Example(features=tf.train.Features(feature=features))


def _parse_example(serialized: bytes) -> dict[str, Any]:
	example = tf.train.Example.FromString(serialized)
	features = example.features.feature

	def text(name: str) -> str:
		return features[name].bytes_list.value[0].decode("utf-8")

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
	dataset = tf.data.TFRecordDataset(str(path), num_parallel_reads=1)
	for serialized in dataset.as_numpy_iterator():
		yield _parse_example(serialized)


def materialize_filtered_split(output_path: Path = RAW_TFRECORD) -> None:
	"""Stream the source once and cache the filtered records as TFRecord."""
	if output_path.exists() and output_path.stat().st_size > 0:
		print(f"Using existing local cache: {output_path}")
		return

	output_path.parent.mkdir(parents=True, exist_ok=True)
	counts: Counter[str] = Counter()
	total = 0
	streamed = load_dataset(DATASET_NAME, split="train", streaming=True)
	with tf.io.TFRecordWriter(str(output_path)) as writer:
		for source_record in streamed:
			if source_record.get("country_code") not in COUNTRIES:
				continue
			record = {
				"image": _image_to_jpeg(source_record["image"]),
				"image_id": source_record["image_id"],
				"panoid": source_record["panoid"],
				"country_code": source_record["country_code"],
				"latitude": float(source_record["latitude"]),
				"longitude": float(source_record["longitude"]),
				"elevation": float(source_record.get("elevation") or 0.0),
			}
			writer.write(_example_from_record(record).SerializeToString())
			counts[record["country_code"]] += 1
			total += 1

	print(f"Materialized {total:,} filtered images to {output_path}")
	print_country_counts(counts)


def print_country_counts(counts: Counter[str]) -> None:
	print("Per-country image counts:")
	for country in sorted(COUNTRIES):
		count = counts[country]
		warning = "  WARNING: under ~2,000 images" if count < 2_000 else ""
		print(f"  {country}: {count:,}{warning}")


def load_cached_records(path: Path = RAW_TFRECORD) -> list[dict[str, Any]]:
	records = list(_iter_records(path))
	print_country_counts(Counter(record["country_code"] for record in records))
	if not records:
		raise RuntimeError(f"No filtered records found in {path}")
	return records


def cluster_geocells(
	records: list[dict[str, Any]],
	centroids_path: Path = CENTROIDS_PATH,
	n_clusters: int = NUM_GEOCELLS,
) -> dict[int, tuple[float, float]]:
	if len(records) < n_clusters:
		raise ValueError(f"Need at least {n_clusters} images for KMeans, found {len(records)}")
	coordinates = np.array(
		[[record["latitude"], record["longitude"]] for record in records], dtype=np.float64
	)
	model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
	labels = model.fit_predict(coordinates)
	for record, label in zip(records, labels):
		record["cell_id"] = int(label)

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


def write_annotated_records(records: list[dict[str, Any]], output_path: Path = PROCESSED_TFRECORD) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with tf.io.TFRecordWriter(str(output_path)) as writer:
		for record in records:
			writer.write(_example_from_record(record, record["cell_id"]).SerializeToString())
	print(f"Wrote annotated records to {output_path}")


def split_by_panoid(
	records: list[dict[str, Any]], dev_fraction: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
	"""Split complete panoids, so views from one location never cross the split."""
	if not 0.0 <= dev_fraction < 1.0:
		raise ValueError("dev_fraction must be in [0, 1)")
	train_records, dev_records = [], []
	for record in records:
		digest = hashlib.sha1(str(record["panoid"]).encode("utf-8")).digest()
		goes_to_dev = int.from_bytes(digest[:8], "big") / 2**64 < dev_fraction
		(dev_records if goes_to_dev else train_records).append(record)
	return train_records, dev_records


def write_split_records(records: list[dict[str, Any]], directory: Path, dev_fraction: float) -> None:
	if dev_fraction == 0.0:
		return
	train_records, dev_records = split_by_panoid(records, dev_fraction)
	write_annotated_records(train_records, directory / "train.tfrecord")
	write_annotated_records(dev_records, directory / "dev.tfrecord")
	print(f"Internal panoid split: {len(train_records):,} train / {len(dev_records):,} dev images")


def _preprocess_image(image_bytes: tf.Tensor, normalization: str, training: bool) -> tf.Tensor:
	image = tf.io.decode_jpeg(image_bytes, channels=3)
	image = tf.image.convert_image_dtype(image, tf.float32)
	if training:
		image = tf.image.resize(image, [232, 232])
		image = tf.image.random_crop(image, [IMAGE_SIZE[0], IMAGE_SIZE[1], 3])
		image = tf.image.random_flip_left_right(image)
	else:
		image = tf.image.resize(image, IMAGE_SIZE)
	if normalization == "minus_one_one":
		image = image * 2.0 - 1.0
	else:
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
	if normalization not in {"minus_one_one", "imagenet"}:
		raise ValueError("normalization must be 'minus_one_one' or 'imagenet'")
	feature_spec = {
		"image": tf.io.FixedLenFeature([], tf.string),
		"cell_id": tf.io.FixedLenFeature([], tf.int64),
	}

	def parse(serialized: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
		parsed = tf.io.parse_single_example(serialized, feature_spec)
		image = _preprocess_image(parsed["image"], normalization, training)
		return image, tf.cast(parsed["cell_id"], tf.int32)

	dataset = tf.data.TFRecordDataset(str(tfrecord_path))
	if shuffle:
		dataset = dataset.shuffle(2_048, reshuffle_each_iteration=True)
	return dataset.map(parse, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(
		tf.data.AUTOTUNE
	)


def build_eval_dataset(
	tfrecord_path: str | Path,
	batch_size: int = 32,
	normalization: str = "minus_one_one",
) -> tf.data.Dataset:
	"""Build an unaugmented pipeline returning images, cell labels, and true coordinates."""
	if normalization not in {"minus_one_one", "imagenet"}:
		raise ValueError("normalization must be 'minus_one_one' or 'imagenet'")
	feature_spec = {
		"image": tf.io.FixedLenFeature([], tf.string),
		"cell_id": tf.io.FixedLenFeature([], tf.int64),
		"latitude": tf.io.FixedLenFeature([], tf.float32),
		"longitude": tf.io.FixedLenFeature([], tf.float32),
	}

	def parse(serialized: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
		parsed = tf.io.parse_single_example(serialized, feature_spec)
		image = _preprocess_image(parsed["image"], normalization, training=False)
		return image, tf.cast(parsed["cell_id"], tf.int32), parsed["latitude"], parsed["longitude"]

	dataset = tf.data.TFRecordDataset(str(tfrecord_path))
	return dataset.map(parse, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(
		tf.data.AUTOTUNE
	)


def print_baseline(records: list[dict[str, Any]]) -> None:
	latitudes = np.array([record["latitude"] for record in records], dtype=np.float64)
	longitudes = np.array([record["longitude"] for record in records], dtype=np.float64)
	centroid_lat, centroid_lon = float(latitudes.mean()), float(longitudes.mean())
	distances = haversine(centroid_lat, centroid_lon, latitudes, longitudes)
	print(
		"Global-centroid baseline: "
		f"mean={float(np.mean(distances)):.2f} km, "
		f"median={float(np.median(distances)):.2f} km"
	)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--dataset", default=DATASET_NAME, help=argparse.SUPPRESS)
	parser.add_argument("--dev-fraction", type=float, default=0.0)
	parser.add_argument("--normalization", choices=("minus_one_one", "imagenet"), default="minus_one_one")
	parser.add_argument("--batch-size", type=int, default=32)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	if args.dataset != DATASET_NAME:
		raise ValueError("This pipeline is defined for the expected World Streetview dataset")
	materialize_filtered_split()
	records = load_cached_records()
	print_baseline(records)
	cluster_geocells(records)
	write_annotated_records(records)
	write_split_records(records, Path("data/processed"), args.dev_fraction)
	print(
		f"Ready: build_tf_dataset('{PROCESSED_TFRECORD}', "
		f"batch_size={args.batch_size}, normalization='{args.normalization}')"
	)


if __name__ == "__main__":
	main()