"""Train a geocell classifier on top of a frozen MobileNetV2 backbone."""

from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf

from data_pipeline_1 import CENTROIDS_PATH, IMAGE_SIZE, PROCESSED_TFRECORD, build_tf_dataset
from geo_utils import load_centroids

CHECKPOINT_DIR = Path("checkpoints")
TRAIN_TFRECORD = Path("data/processed/train.tfrecord")
DEV_TFRECORD = Path("data/processed/dev.tfrecord")


def build_model(num_classes: int, learning_rate: float) -> tuple[tf.keras.Model, tf.keras.Model]:
	"""Return (model, backbone) for a classifier with a frozen MobileNetV2 backbone."""
	backbone = tf.keras.applications.MobileNetV2(
		input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet", pooling="avg"
	)
	backbone.trainable = False

	inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
	x = backbone(inputs, training=False)
	x = tf.keras.layers.Dropout(0.3)(x)
	x = tf.keras.layers.Dense(256, activation="relu")(x)
	x = tf.keras.layers.Dropout(0.3)(x)
	outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

	model = tf.keras.Model(inputs, outputs)
	model.compile(
		optimizer=tf.keras.optimizers.Adam(learning_rate),
		loss="sparse_categorical_crossentropy",
		metrics=["sparse_categorical_accuracy"],
	)
	return model, backbone


def resolve_train_path() -> Path:
	if TRAIN_TFRECORD.exists():
		return TRAIN_TFRECORD
	return PROCESSED_TFRECORD


def build_callbacks(monitor: str) -> list[tf.keras.callbacks.Callback]:
	CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
	return [
		tf.keras.callbacks.ModelCheckpoint(
			str(CHECKPOINT_DIR / "best.keras"), monitor=monitor, save_best_only=True
		),
		tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=5, restore_best_weights=True),
	]


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--epochs", type=int, default=20)
	parser.add_argument("--fine-tune-epochs", type=int, default=10)
	parser.add_argument("--batch-size", type=int, default=32)
	parser.add_argument("--learning-rate", type=float, default=1e-3)
	parser.add_argument("--fine-tune-learning-rate", type=float, default=1e-5)
	parser.add_argument("--unfreeze-layers", type=int, default=30)
	parser.add_argument("--normalization", choices=("minus_one_one", "imagenet"), default="minus_one_one")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	if not CENTROIDS_PATH.exists():
		raise FileNotFoundError(f"{CENTROIDS_PATH} not found; run data_pipeline_1.py first")
	num_classes = len(load_centroids(CENTROIDS_PATH))
	print(f"Training a {num_classes}-way geocell classifier")

	train_path = resolve_train_path()
	print(f"Train data: {train_path}")
	train_ds = build_tf_dataset(
		train_path,
		batch_size=args.batch_size,
		normalization=args.normalization,
		training=True,
		shuffle=True,
	)

	dev_ds = None
	if DEV_TFRECORD.exists():
		print(f"Dev data: {DEV_TFRECORD}")
		dev_ds = build_tf_dataset(
			DEV_TFRECORD,
			batch_size=args.batch_size,
			normalization=args.normalization,
			training=False,
			shuffle=False,
		)
	else:
		print(
			f"No dev split found at {DEV_TFRECORD}; training without validation. "
			"Re-run data_pipeline_1.py with --dev-fraction to enable it."
		)

	monitor = "val_loss" if dev_ds is not None else "loss"
	callbacks = build_callbacks(monitor)

	model, backbone = build_model(num_classes, args.learning_rate)
	model.summary()

	print(f"\nStage 1: training the head for up to {args.epochs} epochs")
	model.fit(train_ds, validation_data=dev_ds, epochs=args.epochs, callbacks=callbacks)

	if args.fine_tune_epochs > 0:
		print(f"\nStage 2: fine-tuning the top {args.unfreeze_layers} backbone layers")
		backbone.trainable = True
		for layer in backbone.layers[: -args.unfreeze_layers]:
			layer.trainable = False
		model.compile(
			optimizer=tf.keras.optimizers.Adam(args.fine_tune_learning_rate),
			loss="sparse_categorical_crossentropy",
			metrics=["sparse_categorical_accuracy"],
		)
		model.fit(train_ds, validation_data=dev_ds, epochs=args.fine_tune_epochs, callbacks=callbacks)

	final_path = CHECKPOINT_DIR / "model.keras"
	model.save(final_path)
	print(f"\nSaved final model to {final_path}")
	if (CHECKPOINT_DIR / "best.keras").exists():
		print(f"Best checkpoint (by {monitor}) saved to {CHECKPOINT_DIR / 'best.keras'}")


if __name__ == "__main__":
	main()
