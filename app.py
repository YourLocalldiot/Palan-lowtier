"""Streamlit app: guess where a street-view style photo was taken."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pillow_avif  # noqa: F401 - registers AVIF support with Pillow
import pydeck as pdk
import streamlit as st
import tensorflow as tf
from PIL import Image

from geo_utils import weighted_centroid

EXPORT_DIR = Path(__file__).resolve().parent / "export"
MODEL_PATH = EXPORT_DIR / "model.tflite"
CENTROIDS_PATH = EXPORT_DIR / "cell_centroids.json"
CONFIG_PATH = EXPORT_DIR / "config.json"
TOP_K = 5

# Dark, high-contrast theme inspired by palantir.com's measured styles: near-black
# background, off-white/muted-gray text, sharp 0px-radius corners, and a regular-weight
# display font with tight letter-spacing. Palantir's actual typeface ("Alliance No.1/2")
# is proprietary/licensed, so Inter stands in as a free lookalike for the same feel.
STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"], .stApp {
	font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important;
}

h1, h2, h3 {
	font-weight: 400 !important;
	letter-spacing: -0.03em !important;
}

h1 { font-size: 2.75rem !important; }

p, .stCaption, [data-testid="stCaptionContainer"] {
	color: #B9B9B9 !important;
}

/* Sharp corners everywhere, no rounded UI elements */
button, input, textarea, select,
[data-testid="stFileUploaderDropzone"],
[data-testid="stFileUploader"] section,
[data-testid="stMetric"],
[data-testid="stAlert"],
[data-testid="stImage"] img,
.stButton > button,
div[data-baseweb="select"] > div {
	border-radius: 0px !important;
}

/* Primary CTA-style buttons: white on near-black, matching Palantir's "Get Started" */
.stButton > button, [data-testid="stFileUploader"] button {
	background-color: #EFEFEF !important;
	color: #0D0E10 !important;
	border: none !important;
	font-weight: 400 !important;
	letter-spacing: normal !important;
}
.stButton > button:hover, [data-testid="stFileUploader"] button:hover {
	background-color: #FFFFFF !important;
}

/* Metric labels: tracked uppercase; values in monospace for a technical readout feel */
[data-testid="stMetricLabel"] {
	text-transform: uppercase !important;
	letter-spacing: 0.08em !important;
	font-size: 0.72rem !important;
	color: #8A8D91 !important;
}
[data-testid="stMetricValue"] {
	font-family: 'JetBrains Mono', 'IBM Plex Mono', monospace !important;
	font-weight: 500 !important;
}

hr {
	border-color: rgba(255, 255, 255, 0.1) !important;
}

[data-testid="stFileUploaderDropzone"] {
	background-color: #16181B !important;
	border: 1px solid rgba(255, 255, 255, 0.15) !important;
}
</style>
"""


@st.cache_resource
def load_interpreter() -> tf.lite.Interpreter:
	interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
	interpreter.allocate_tensors()
	return interpreter


@st.cache_resource
def load_metadata() -> tuple[dict[int, tuple[float, float]], dict]:
	centroids_raw = json.loads(CENTROIDS_PATH.read_text(encoding="utf-8"))
	centroids = {
		int(cell_id): (value["centroid_lat"], value["centroid_lon"])
		for cell_id, value in centroids_raw.items()
	}
	config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
	return centroids, config


def preprocess(image: Image.Image, image_size: tuple[int, int], normalization: str) -> np.ndarray:
	resized = image.convert("RGB").resize(image_size)
	array = np.asarray(resized, dtype=np.float32) / 255.0
	if normalization == "minus_one_one":
		array = array * 2.0 - 1.0
	else:
		mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
		std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
		array = (array - mean) / std
	return array[np.newaxis, ...]


def predict(interpreter: tf.lite.Interpreter, batch: np.ndarray) -> np.ndarray:
	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()
	interpreter.set_tensor(input_details[0]["index"], batch)
	interpreter.invoke()
	return interpreter.get_tensor(output_details[0]["index"])[0]


def main() -> None:
	st.set_page_config(page_title="Palan-lowtier", page_icon="\U0001f30f")
	st.markdown(STYLE, unsafe_allow_html=True)
	st.title("Palan Lowtier: Guess the Location")
	st.caption("Currently trained on street-view images from across Asia.")

	if not MODEL_PATH.exists() or not CENTROIDS_PATH.exists() or not CONFIG_PATH.exists():
		st.error(f"No exported model bundle found in {EXPORT_DIR}/. Run export_4.py first.")
		return

	interpreter = load_interpreter()
	centroids, config = load_metadata()
	image_size = tuple(config["image_size"])
	normalization = config["normalization"]

	uploaded_file = st.file_uploader(
		"Upload a street-view style photo",
		type=["jpg", "jpeg", "png", "webp", "tiff", "tif", "jfif", "avif"],
	)
	if uploaded_file is None:
		st.info("Upload an image to get a coordinate guess.")
		return

	image = Image.open(uploaded_file)
	st.image(image, caption="Uploaded image", use_container_width=True)

	batch = preprocess(image, image_size, normalization)
	probabilities = predict(interpreter, batch)

	num_classes = len(centroids)
	centroid_lat = np.array([centroids[i][0] for i in range(num_classes)])
	centroid_lon = np.array([centroids[i][1] for i in range(num_classes)])

	top_indices = np.argsort(probabilities)[::-1][:3]
	best_cell = int(top_indices[0])
	confidence = float(probabilities[best_cell])
	guess_lat, guess_lon = weighted_centroid(probabilities, centroid_lat, centroid_lon, top_k=TOP_K)

	st.subheader("Best guess")
	st.caption(f"Weighted average of the top {TOP_K} predicted cells")
	col1, col2, col3 = st.columns(3)
	col1.metric("Latitude", f"{guess_lat:.4f}")
	col2.metric("Longitude", f"{guess_lon:.4f}")
	col3.metric("Top cell confidence", f"{confidence:.1%}")

	st.pydeck_chart(
		pdk.Deck(
			map_style=None,
			initial_view_state=pdk.ViewState(latitude=guess_lat, longitude=guess_lon, zoom=4),
			layers=[
				pdk.Layer(
					"ScatterplotLayer",
					data=[{"lat": guess_lat, "lon": guess_lon}],
					get_position="[lon, lat]",
					get_fill_color=[220, 40, 40],
					get_radius=40000,
				)
			],
		)
	)

	st.subheader("Top 3 predictions")
	for rank, cell_id in enumerate(top_indices, start=1):
		lat, lon = centroids[int(cell_id)]
		st.write(f"{rank}. cell {cell_id}: ({lat:.4f}, {lon:.4f}) — {probabilities[cell_id]:.1%}")


if __name__ == "__main__":
	main()
