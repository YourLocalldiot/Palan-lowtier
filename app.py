"""Streamlit app: guess where a street-view style photo was taken."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pillow_avif  # noqa: F401 - registers AVIF support with Pillow
import pycountry
import pydeck as pdk
import streamlit as st
import tensorflow as tf
from google import genai
from PIL import Image

from geo_utils import reverse_geocode as _reverse_geocode_raw
from geo_utils import weighted_centroid

PALANTIR_URL = "https://www.palantir.com"
GEMINI_MODEL = "gemini-flash-latest"

EXPORT_DIR = Path(__file__).resolve().parent / "export"
MODEL_PATH = EXPORT_DIR / "model.tflite"
CENTROIDS_PATH = EXPORT_DIR / "cell_centroids.json"
CONFIG_PATH = EXPORT_DIR / "config.json"
COUNTRY_ACCURACY_PATH = EXPORT_DIR / "country_accuracy.json"
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

.st-key-navbar {
	justify-content: space-between !important;
	width: 100% !important;
	padding: 0 0 1rem 0;
	margin-bottom: 1.5rem;
	border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}
.pt-navbar-left {
	display: flex;
	align-items: center;
	gap: 1.5rem;
}
.pt-nav-wordmark {
	font-size: 1.05rem;
	font-weight: 500;
	letter-spacing: -0.01em;
	color: #EFEFEF !important;
	text-decoration: none !important;
}
.pt-nav-link, [data-testid="stPageLink-NavLink"] {
	font-size: 0.72rem !important;
	text-transform: uppercase !important;
	letter-spacing: 0.08em !important;
	color: #8A8D91 !important;
	text-decoration: none !important;
	border-bottom: 1px solid rgba(255, 255, 255, 0.25) !important;
	padding-bottom: 2px !important;
	white-space: nowrap;
	background: transparent !important;
}
.pt-nav-link:hover, [data-testid="stPageLink-NavLink"]:hover {
	color: #EFEFEF !important;
	border-color: #EFEFEF !important;
	background: transparent !important;
}
[data-testid="stPageLink-NavLink"] p {
	font-size: inherit !important;
	color: inherit !important;
}

.location-result {
	font-size: 1.6rem;
	font-weight: 400;
	letter-spacing: -0.02em;
	color: #EFEFEF;
	margin: 0.25rem 0 1rem 0;
}
.location-result .country {
	color: #8A8D91;
}
</style>
"""

NAVBAR_LEFT = f"""
<div class="pt-navbar-left">
	<a class="pt-nav-wordmark" href="{PALANTIR_URL}" target="_blank" rel="noopener noreferrer">Palantir</a>
	<a class="pt-nav-link" href="{PALANTIR_URL}" target="_blank" rel="noopener noreferrer">&larr; Palantir this way</a>
</div>
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


@st.cache_data(show_spinner=False, ttl=3600)
def reverse_geocode(lat: float, lon: float) -> tuple[str, str]:
	"""Return (region, country) for a coordinate. Thin cached wrapper around geo_utils'
	shared implementation, kept here so app.py's Streamlit session cache applies."""
	result = _reverse_geocode_raw(lat, lon)
	return result["region"], result["country_name"]


def _normalize_country(name: str) -> str:
	return "".join(name.lower().split())


def _country_matches(a: str, b: str) -> bool:
	na, nb = _normalize_country(a), _normalize_country(b)
	return bool(na) and bool(nb) and (na == nb or na in nb or nb in na)


@st.cache_resource
def load_gemini_client() -> genai.Client | None:
	"""Return a Gemini client if GEMINI_API_KEY is configured, else None.

	Missing configuration is not an error: the app works fine without it, just
	without the cross-check fallback for country-ambiguous predictions.
	"""
	try:
		api_key = st.secrets["GEMINI_API_KEY"]
	except Exception:
		return None
	if not api_key:
		return None
	return genai.Client(api_key=api_key)


def gemini_guess_country(client: genai.Client, image: Image.Image, candidate_countries: list[str]) -> str | None:
	"""Ask Gemini to pick the most likely country from candidates, given the photo.

	Only meant to break ties when the CNN's own top predictions already span more
	than one of these candidates (see main()) - not a general-purpose classifier.
	Returns None on any failure or an answer that doesn't match a candidate.
	"""
	prompt = (
		"This photo is a street-view style image. Based on visual cues (text/language on "
		"signs, license plates, which side vehicles drive on, architecture, vegetation), "
		f"which of these countries is it most likely from: {', '.join(candidate_countries)}? "
		"Reply with ONLY the single most likely country name from that list, nothing else."
	)
	try:
		response = client.models.generate_content(model=GEMINI_MODEL, contents=[prompt, image])
		answer = (response.text or "").strip()
	except Exception:
		return None
	for country in candidate_countries:
		if _country_matches(answer, country):
			return country
	return None


def home_page() -> None:
	st.title("Palan-lowtier: Guess the Location")
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

	top_k_indices = np.argsort(probabilities)[::-1][:TOP_K]
	best_cell = int(top_k_indices[0])
	confidence = float(probabilities[best_cell])
	guess_lat, guess_lon = weighted_centroid(probabilities, centroid_lat, centroid_lon, top_k=TOP_K)

	st.subheader("Best guess")
	st.caption(f"Weighted average of the top {TOP_K} predicted cells")

	gemini_note = None
	with st.spinner("Looking up the nearest region and country..."):
		# Reverse-geocode each of the CNN's top-k cells once, up front. This tells us
		# whether the model's own candidates span more than one country (a direct
		# sign of cross-country ambiguity the plain weighted-centroid blend can't
		# resolve), and doubles as a ranked fallback chain for display below.
		cell_geocodes = [reverse_geocode(*centroids[int(idx)]) for idx in top_k_indices]
		cell_countries = [country for _, country in cell_geocodes]
		distinct_countries = sorted({c for c in cell_countries if c})

	if len(distinct_countries) > 1:
		gemini_client = load_gemini_client()
		if gemini_client is not None:
			with st.spinner("Cross-checking with Gemini..."):
				chosen_country = gemini_guess_country(gemini_client, image, distinct_countries)
			if chosen_country is not None:
				matching = [
					idx for idx, c in zip(top_k_indices, cell_countries) if _country_matches(c, chosen_country)
				]
				guess_lat, guess_lon = weighted_centroid(
					probabilities[matching], centroid_lat[matching], centroid_lon[matching],
					top_k=len(matching), max_distance_km=float("inf"),
				)
				gemini_note = f"Model's top guesses spanned {', '.join(distinct_countries)} - Gemini narrowed it to {chosen_country}."
		else:
			gemini_note = (
				f"Model's top guesses spanned {', '.join(distinct_countries)}, but no GEMINI_API_KEY is "
				"configured to break the tie - showing the plain weighted-average guess."
			)

	region, country = reverse_geocode(guess_lat, guess_lon)
	used_fallback_rank = None
	if not country:
		# The blended point - or even a single geocell's own centroid, for
		# archipelago countries where K-means can average several islands into an
		# open-water point - can fall outside any mapped border. Walk the ranked
		# candidates and use the first one that actually resolves, rather than
		# giving up after just one attempt.
		for rank, (candidate_region, candidate_country) in enumerate(cell_geocodes, start=1):
			if candidate_country:
				region, country = candidate_region, candidate_country
				used_fallback_rank = rank
				break

	if country:
		location_label = f"{region}, <span class='country'>{country}</span>" if region else country
	else:
		location_label = "Unable to determine region/country right now"
	st.markdown(f"<div class='location-result'>{location_label}</div>", unsafe_allow_html=True)
	if used_fallback_rank is not None:
		st.caption(
			"The blended guess falls outside any mapped border (likely open water) - "
			f"showing rank {used_fallback_rank}'s predicted location instead."
		)
	if gemini_note:
		st.caption(gemini_note)

	col1, col2, col3 = st.columns(3)
	col1.metric("Latitude", f"{guess_lat:.4f}")
	col2.metric("Longitude", f"{guess_lon:.4f}")
	col3.metric("Top cell confidence", f"{confidence:.1%}")

	top_cells_data = [
		{
			"lat": float(centroids[int(cell_id)][0]),
			"lon": float(centroids[int(cell_id)][1]),
			"label": f"Rank {rank}: cell {int(cell_id)} ({probabilities[cell_id]:.1%})",
			# Higher-confidence cells get a visibly bigger ring, so confidence is
			# readable from the map itself, not just the tooltip.
			"radius": 12_000 + float(probabilities[cell_id]) * 70_000,
		}
		for rank, cell_id in enumerate(top_k_indices, start=1)
	]

	st.pydeck_chart(
		pdk.Deck(
			map_style=None,
			initial_view_state=pdk.ViewState(latitude=guess_lat, longitude=guess_lon, zoom=4),
			tooltip={"text": "{label}"},
			layers=[
				pdk.Layer(
					"ScatterplotLayer",
					data=top_cells_data,
					get_position="[lon, lat]",
					get_radius="radius",
					get_fill_color=[239, 159, 39, 70],
					get_line_color=[239, 159, 39, 255],
					stroked=True,
					get_line_width=1500,
					pickable=True,
				),
				pdk.Layer(
					"ScatterplotLayer",
					data=[{"lat": guess_lat, "lon": guess_lon, "label": f"Best guess ({confidence:.1%} top cell)"}],
					get_position="[lon, lat]",
					get_fill_color=[220, 40, 40],
					get_radius=40000,
					pickable=True,
				),
			],
		)
	)
	st.caption(
		"Red = final blended guess. Amber rings = top 5 predicted cells "
		"(bigger ring = higher confidence - hover for rank and cell details)."
	)

	st.subheader(f"Top {TOP_K} predictions")
	for rank, cell_id in enumerate(top_k_indices, start=1):
		lat, lon = centroids[int(cell_id)]
		st.write(f"{rank}. cell {cell_id}: ({lat:.4f}, {lon:.4f}) — {probabilities[cell_id]:.1%}")


def model_stats_page() -> None:
	st.title("Model Stats")
	st.caption("Per-country accuracy on the held-out dev set, from the most recent evaluate_3.py run.")

	if not COUNTRY_ACCURACY_PATH.exists():
		st.info(
			"No country-accuracy data yet. Run evaluate_3.py (after data_pipeline_1.py and "
			"train_2.py) and then export_4.py to generate it."
		)
		return

	per_country = json.loads(COUNTRY_ACCURACY_PATH.read_text(encoding="utf-8"))

	rows = []
	for country in pycountry.countries:
		code = country.alpha_2
		stats = per_country.get(code)
		if stats and stats.get("total", 0) > 0:
			rows.append(
				{
					"Country": country.name,
					"Accuracy": stats["correct"] / stats["total"],
					"Dev images": f"{stats['total']:,}",
				}
			)
		else:
			rows.append({"Country": country.name, "Accuracy": None, "Dev images": ""})

	table = pd.DataFrame(rows).sort_values(
		by=["Accuracy", "Country"], ascending=[False, True], na_position="last"
	)

	available = table["Accuracy"].notna().sum()
	st.caption(f"{available} of {len(table)} countries/territories have dev-set data.")

	st.dataframe(
		table,
		use_container_width=True,
		hide_index=True,
		height=600,
		column_config={
			"Accuracy": st.column_config.ProgressColumn(
				"Accuracy", format="percent", min_value=0.0, max_value=1.0
			),
		},
	)


def main() -> None:
	st.set_page_config(page_title="Palan-lowtier", page_icon="\U0001f30f")
	st.markdown(STYLE, unsafe_allow_html=True)

	home = st.Page(home_page, title="Home", url_path="home", default=True)
	stats = st.Page(model_stats_page, title="Model Stats", url_path="model-stats")
	page = st.navigation([home, stats], position="hidden")

	with st.container(horizontal=True, vertical_alignment="center", key="navbar"):
		st.markdown(NAVBAR_LEFT, unsafe_allow_html=True)
		if page is home:
			st.page_link(stats, label="Model Stats →")
		else:
			st.page_link(home, label="← Home")

	page.run()


if __name__ == "__main__":
	main()
