"""Train a geocell classifier on top of a frozen MobileNetV2 backbone."""

# =============================================================================
# train_2.py - STEP 2 of 5: train the neural network
# -----------------------------------------------------------------------------
# Input : the labelled TFRecord files and cell_centroids.json produced by
#         data_pipeline_1.py
# Output: checkpoints/best.keras  (the best version seen during training)
#         checkpoints/model.keras (the version at the very end of training)
#
# THE TASK
#   Given a photo, output a confidence score for each geocell ("62% geocell A,
#   29% geocell C, 9% geocell B, ..."). This is ordinary image classification,
#   where each geocell is one class.
#
# THE APPROACH: TRANSFER LEARNING
#   Training an image model from scratch needs millions of images and a GPU.
#   Instead we reuse MobileNetV2, a network that Google already trained on
#   ImageNet (1.2 million everyday photos). It already knows how to detect
#   edges, textures, shapes and objects. We keep that knowledge (the
#   "backbone") and add a small new "head" on top that learns to map those
#   visual features to our geocells.
#
#   Training happens in two stages:
#     Stage 1 - backbone FROZEN; only the new head learns. Fast and safe.
#     Stage 2 - "fine-tuning": unfreeze the top layers of the backbone and let
#               them adjust slightly (very small learning rate) so they become
#               more sensitive to geography clues: signs, road markings,
#               vegetation, architecture.
#
# Typical usage (from a terminal):
#     python train_2.py --epochs 3 --fine-tune-epochs 5
# =============================================================================

from __future__ import annotations

import argparse            # read command-line flags like --epochs
from pathlib import Path   # file paths

import tensorflow as tf    # the deep-learning framework (Keras is part of it)

# Shared settings and the data-loading pipeline from step 1.
from data_pipeline_1 import CENTROIDS_PATH, IMAGE_SIZE, PROCESSED_TFRECORD, build_tf_dataset
from geo_utils import load_centroids

# Folder where trained models are saved.
CHECKPOINT_DIR = Path("checkpoints")
# Written by data_pipeline_1.py only when run with --dev-fraction.
TRAIN_TFRECORD = Path("data/processed/train.tfrecord")
DEV_TFRECORD = Path("data/processed/dev.tfrecord")


def build_model(num_classes: int, learning_rate: float) -> tuple[tf.keras.Model, tf.keras.Model]:
	"""Return (model, backbone) for a classifier with a frozen MobileNetV2 backbone."""
	# ---- The backbone: pretrained MobileNetV2 -------------------------------
	#   include_top=False  -> drop MobileNetV2's original last layer, which
	#                         predicts ImageNet's 1,000 object categories
	#                         (dog, car, ...). We need our geocells instead.
	#   weights="imagenet" -> load the knowledge learned from ImageNet
	#   pooling="avg"      -> squash the backbone's output into one flat list of
	#                         1,280 numbers per image (a "feature vector" that
	#                         summarises what the image contains)
	backbone = tf.keras.applications.MobileNetV2(
		input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet", pooling="avg"
	)
	# Freeze the backbone: its weights will NOT change during Stage 1.
	backbone.trainable = False

	# ---- The head: new layers that we train ---------------------------------
	# Input: a 224 x 224 image with 3 colour channels (RGB).
	inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
	# training=False keeps the backbone's batch-normalization layers in
	# "inference mode" so their stored statistics aren't disturbed.
	x = backbone(inputs, training=False)
	# Dropout randomly switches off 30% of values during training. This stops the
	# head relying too heavily on any single feature, which reduces overfitting
	# (memorising the training photos instead of learning general patterns).
	x = tf.keras.layers.Dropout(0.3)(x)
	# A fully connected layer of 256 neurons that learns combinations of the
	# backbone's features. "relu" lets it learn non-linear patterns.
	x = tf.keras.layers.Dense(256, activation="relu")(x)
	x = tf.keras.layers.Dropout(0.3)(x)
	# Output layer: one neuron per geocell. "softmax" turns the raw scores into
	# probabilities that are all between 0 and 1 and add up to 1 (100%).
	outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

	model = tf.keras.Model(inputs, outputs)
	# compile() chooses HOW the model learns:
	#   optimizer = Adam: the algorithm that nudges the weights after each batch;
	#               learning_rate controls how big each nudge is.
	#   loss      = sparse categorical cross-entropy: the standard "how wrong
	#               was it?" score for picking one class out of many. "Sparse"
	#               means labels are plain geocell numbers (e.g. 47) rather than
	#               long one-hot lists [0, 0, ..., 1, ..., 0].
	#   metrics   = accuracy: the % of photos whose top-1 geocell was exactly right.
	model.compile(
		optimizer=tf.keras.optimizers.Adam(learning_rate),
		loss="sparse_categorical_crossentropy",
		metrics=["sparse_categorical_accuracy"],
	)
	# The backbone is returned too, so Stage 2 can unfreeze part of it later.
	return model, backbone


def resolve_train_path() -> Path:
	# Prefer the 80% training split. If data_pipeline_1.py was run WITHOUT
	# --dev-fraction, only the combined file exists, so fall back to training on
	# all of it (meaning there is nothing left over for fair evaluation).
	if TRAIN_TFRECORD.exists():
		return TRAIN_TFRECORD
	return PROCESSED_TFRECORD


def build_callbacks(monitor: str) -> list[tf.keras.callbacks.Callback]:
	# Callbacks are hooks that Keras runs automatically at the end of each epoch.
	# `monitor` is the number they watch: "val_loss" (error on the dev set) when
	# a dev set exists, otherwise "loss" (error on the training set).
	CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
	return [
		# Save the model to checkpoints/best.keras - but only when the monitored
		# number improves, so the file always holds the best version so far.
		tf.keras.callbacks.ModelCheckpoint(
			str(CHECKPOINT_DIR / "best.keras"), monitor=monitor, save_best_only=True
		),
		# Stop training early if the monitored number hasn't improved for 5
		# epochs in a row, and roll back to the best weights seen. This saves
		# time and stops the model overfitting.
		tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=5, restore_best_weights=True),
	]


def parse_args() -> argparse.Namespace:
	# Optional command-line flags, e.g. python train_2.py --epochs 3
	parser = argparse.ArgumentParser(description=__doc__)
	# Maximum passes over the training data in Stage 1 (early stopping may end sooner).
	parser.add_argument("--epochs", type=int, default=20)
	# Maximum passes in Stage 2 (fine-tuning). 0 skips Stage 2 entirely.
	parser.add_argument("--fine-tune-epochs", type=int, default=10)
	# Photos processed together before each weight update.
	parser.add_argument("--batch-size", type=int, default=32)
	# Step size for Stage 1: relatively large, because the head starts from scratch.
	parser.add_argument("--learning-rate", type=float, default=1e-3)
	# Step size for Stage 2: 100x smaller, so the pretrained backbone is only
	# gently adjusted instead of having its existing knowledge overwritten.
	parser.add_argument("--fine-tune-learning-rate", type=float, default=1e-5)
	# How many of the backbone's LAST layers to unfreeze in Stage 2. The last
	# layers detect high-level patterns (the most task-specific ones); the early
	# layers detect generic edges/colours and are fine as they are.
	parser.add_argument("--unfreeze-layers", type=int, default=30)
	# Must match how images were normalised (see data_pipeline_1.py).
	parser.add_argument("--normalization", choices=("minus_one_one", "imagenet"), default="minus_one_one")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	if not CENTROIDS_PATH.exists():
		raise FileNotFoundError(f"{CENTROIDS_PATH} not found; run data_pipeline.py first")
	# The number of classes is however many geocells step 1 created, so the
	# model automatically matches whatever --num-geocells was used.
	num_classes = len(load_centroids(CENTROIDS_PATH))
	print(f"Training a {num_classes}-way geocell classifier")

	# ---- Load the training data (with augmentation and shuffling) -----------
	train_path = resolve_train_path()
	print(f"Train data: {train_path}")
	train_ds = build_tf_dataset(
		train_path,
		batch_size=args.batch_size,
		normalization=args.normalization,
		training=True,
		shuffle=True,
	)

	# ---- Load the dev data, if a split exists -------------------------------
	# The dev set is only used to MEASURE the model after each epoch, never to
	# train it, so it shows how well the model handles photos it hasn't seen.
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
			"Re-run data_pipeline.py with --dev-fraction to enable it."
		)

	# Watch dev-set error when available (the honest measure); otherwise the
	# training error is the only thing available to watch.
	monitor = "val_loss" if dev_ds is not None else "loss"
	callbacks = build_callbacks(monitor)

	model, backbone = build_model(num_classes, args.learning_rate)
	# Print a table of every layer and how many weights it has.
	model.summary()

	# ---- Stage 1: train only the new head -----------------------------------
	# fit() loops over the training data `epochs` times. Each loop is one epoch;
	# each epoch is made of many steps, one step per batch of photos.
	print(f"\nStage 1: training the head for up to {args.epochs} epochs")
	model.fit(train_ds, validation_data=dev_ds, epochs=args.epochs, callbacks=callbacks)

	# ---- Stage 2: fine-tune the top of the backbone -------------------------
	if args.fine_tune_epochs > 0:
		print(f"\nStage 2: fine-tuning the top {args.unfreeze_layers} backbone layers")
		# Unfreeze the whole backbone...
		backbone.trainable = True
		# ...then re-freeze everything EXCEPT the last `unfreeze_layers` layers.
		# (backbone.layers[:-30] means "every layer except the last 30".)
		for layer in backbone.layers[: -args.unfreeze_layers]:
			layer.trainable = False
		# Keras only notices the change in which layers are trainable after
		# compile() is called again. This is also where the much smaller
		# fine-tuning learning rate is applied.
		model.compile(
			optimizer=tf.keras.optimizers.Adam(args.fine_tune_learning_rate),
			loss="sparse_categorical_crossentropy",
			metrics=["sparse_categorical_accuracy"],
		)
		# Same data and callbacks, so best.keras keeps tracking the overall best.
		model.fit(train_ds, validation_data=dev_ds, epochs=args.fine_tune_epochs, callbacks=callbacks)

	# ---- Save the final model -----------------------------------------------
	# model.keras = the model as it is at the end. best.keras (saved by the
	# checkpoint callback) = the best-scoring version, which is the one the
	# later steps (evaluate_3.py, export_4.py) use by default.
	final_path = CHECKPOINT_DIR / "model.keras"
	model.save(final_path)
	print(f"\nSaved final model to {final_path}")
	if (CHECKPOINT_DIR / "best.keras").exists():
		print(f"Best checkpoint (by {monitor}) saved to {CHECKPOINT_DIR / 'best.keras'}")


# Run main() only when this file is executed directly (python train_2.py).
if __name__ == "__main__":
	main()
