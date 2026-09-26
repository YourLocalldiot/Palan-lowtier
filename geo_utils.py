"""Geospatial helpers shared by the project scripts."""

# =============================================================================
# geo_utils.py - shared geography and math helpers
# -----------------------------------------------------------------------------
# This file is a small "toolbox" of functions that several other files need:
#
#   haversine()              -> distance in km between two points on Earth
#   load_centroids()         -> read the geocell centre points from JSON
#   accuracy_at_thresholds() -> "what % of guesses landed within X km?"
#   weighted_centroid()      -> turn the model's probabilities into ONE lat/lon
#   reverse_geocode()        -> turn a lat/lon into a region + country name
#
# It is imported by data_pipeline_1.py, evaluate_3.py, export_4.py and app.py.
# Keeping this logic in one place means every script calculates distances and
# locations exactly the same way, instead of each having its own copy.
# =============================================================================

# Allows modern type hints such as `float | np.ndarray` to be written freely.
from __future__ import annotations

import json                  # read the cell_centroids.json file
from pathlib import Path     # file paths that work on both Windows and Linux

import numpy as np           # fast maths on whole arrays of numbers at once
import requests              # make HTTP calls to the reverse-geocoding web API


# Mean radius of the Earth in kilometres. The haversine formula first works out
# the ANGLE between two points around the globe; multiplying that angle by the
# Earth's radius converts it into an actual distance in km.
EARTH_RADIUS_KM = 6371.0088

# Standard geolocation error buckets (street / city / region / country / continent).
# These are the same thresholds used in published geolocation research, so our
# results can be compared against other models:
#     1 km    ~ correct street
#     25 km   ~ correct city
#     200 km  ~ correct region / province
#     750 km  ~ correct country
#     2500 km ~ correct continent
ACCURACY_THRESHOLDS_KM = (1.0, 25.0, 200.0, 750.0, 2500.0)

# A free "reverse geocoding" web API: send it a latitude/longitude and it replies
# with the region and country that point sits inside. No API key is required.
REVERSE_GEOCODE_URL = "https://api.bigdatacloud.net/data/reverse-geocode-client"


def haversine(
	lat1: float | np.ndarray,
	lon1: float | np.ndarray,
	lat2: float | np.ndarray,
	lon2: float | np.ndarray,
) -> float | np.ndarray:
	"""Return great-circle distance in kilometres between coordinate pairs."""
	# The haversine formula measures the "great-circle" distance: the shortest
	# route between two points along the curved surface of the Earth (like an
	# airplane's flight path), not a straight line tunnelling through the planet.
	#
	# Every argument can be a single number OR a numpy array. That means the same
	# function can measure one distance, or millions of distances in one call,
	# without writing a slow Python loop.

	# numpy's sin/cos work in radians, but GPS coordinates are in degrees,
	# so convert all four inputs first.
	lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(
		np.radians, (lat1, lon1, lat2, lon2)
	)
	# How far apart the two points are north-south and east-west (in radians).
	delta_lat = lat2_rad - lat1_rad
	delta_lon = lon2_rad - lon1_rad
	# The core of the haversine formula. The result is a number from 0 to 1 that
	# describes how far apart the points are around the sphere
	# (0 = the same spot, 1 = exact opposite sides of the Earth).
	haversine_angle = (
		np.sin(delta_lat / 2.0) ** 2
		+ np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2.0) ** 2
	)
	# Convert that value into a central angle with arcsin, then multiply by the
	# Earth's radius to get kilometres. np.clip keeps the value inside [0, 1]:
	# tiny floating-point rounding errors could otherwise push it to 1.0000001,
	# which would make sqrt/arcsin return NaN ("not a number").
	distance = 2.0 * EARTH_RADIUS_KM * np.arcsin(
		np.sqrt(np.clip(haversine_angle, 0.0, 1.0))
	)
	# If single numbers were passed in, hand back a plain Python float.
	# If arrays were passed in, hand back the whole array of distances.
	return float(distance) if np.ndim(distance) == 0 else distance


def load_centroids(path: Path) -> dict[int, tuple[float, float]]:
	"""Load a cell_centroids.json mapping cell id to (lat, lon)."""
	# cell_centroids.json is written by data_pipeline_1.py and looks like:
	#     { "0": {"centroid_lat": 21.03, "centroid_lon": 105.85}, "1": {...}, ... }
	# JSON object keys are always strings, so the cell id is converted back to an
	# int here. The result is a simple lookup: {0: (21.03, 105.85), 1: (...), ...}
	# meaning "geocell number N is centred at this latitude/longitude".
	data = json.loads(Path(path).read_text(encoding="utf-8"))
	return {int(cell_id): (value["centroid_lat"], value["centroid_lon"]) for cell_id, value in data.items()}


def accuracy_at_thresholds(
	distances_km: np.ndarray, thresholds_km: tuple[float, ...] = ACCURACY_THRESHOLDS_KM
) -> dict[float, float]:
	"""Return the fraction of distances at or under each threshold, in kilometres."""
	# distances_km holds one number per test image: how far (in km) the model's
	# guess was from the photo's true location.
	distances_km = np.asarray(distances_km, dtype=np.float64)
	# For each threshold, `distances_km <= threshold` produces an array of
	# True/False values. Taking the mean of True/False treats True as 1 and
	# False as 0, so the mean IS the fraction of guesses within that distance.
	# Example result: {1.0: 0.00, 25.0: 0.03, 200.0: 0.28, 750.0: 0.66, 2500.0: 0.93}
	return {threshold: float(np.mean(distances_km <= threshold)) for threshold in thresholds_km}


def weighted_centroid(
	probabilities: np.ndarray,
	centroid_lat: np.ndarray,
	centroid_lon: np.ndarray,
	top_k: int = 5,
	max_distance_km: float = 800.0,
) -> tuple[np.ndarray, np.ndarray] | tuple[float, float]:
	"""Average the top-k predicted cells' centroids, weighted by renormalized probability.

	Smooths over near-miss cells instead of committing to a single argmax cell, which
	reduces distance error when the single most-likely cell is a nearby miss. Cells
	farther than max_distance_km from the single most-likely (top-1) cell are dropped
	before averaging: without this, a model uncertain *between* widely separated regions
	(e.g. different countries) would have its top-k centroids blended into a meaningless
	midpoint (e.g. the ocean between them) rather than falling back to its single best
	guess. Accepts a single probability vector or a batch of them; returns a matching shape.
	"""
	# INPUTS
	#   probabilities : the model's output. One confidence score per geocell,
	#                   e.g. [0.62, 0.09, 0.29, ...]. Can be one photo (1-D array)
	#                   or a batch of photos (2-D array: photos x geocells).
	#   centroid_lat / centroid_lon : the centre point of every geocell, lined up
	#                   so that index i in these arrays matches geocell i.
	#
	# IDEA
	#   Instead of trusting only the single most-confident geocell, take the top
	#   few and average their locations, weighting each by how confident the model
	#   was. Example: 62% at latitude 20, 9% at latitude 60, 29% at latitude 0
	#   -> 0.62*20 + 0.09*60 + 0.29*0 = 17.8.
	#
	# SAFEGUARD
	#   Averaging only makes sense for places that are close together. Averaging
	#   Vietnam and Japan would land in the sea between them, so any candidate
	#   more than max_distance_km from the top guess is ignored.

	probabilities = np.asarray(probabilities, dtype=np.float64)
	# The code below always works on a 2-D table (photos x geocells). If we were
	# given just one photo's probabilities, wrap it as a table with one row, and
	# remember to unwrap the answer again at the end.
	single = probabilities.ndim == 1
	if single:
		probabilities = probabilities[np.newaxis, :]

	# Never ask for more candidates than there are geocells.
	k = min(top_k, probabilities.shape[1])
	# Find the k most confident geocells for each photo:
	#   argsort      -> geocell indices ordered from LEAST to MOST confident
	#   [:, ::-1]    -> reverse, so the MOST confident comes first
	#   [:, :k]      -> keep only the first k
	top_indices = np.argsort(probabilities, axis=1)[:, ::-1][:, :k]
	# Look up the actual probability values for those k geocells.
	top_probs = np.take_along_axis(probabilities, top_indices, axis=1)

	# Location of the single most-confident (top-1) geocell for each photo.
	# `[:, :1]` (instead of `[:, 0]`) keeps the result 2-D, so numpy can compare
	# it against all k candidates at once (this is called "broadcasting").
	top1_lat = centroid_lat[top_indices[:, :1]]
	top1_lon = centroid_lon[top_indices[:, :1]]
	# How far (km) each of the k candidates is from the top-1 guess.
	distance_from_top1 = haversine(top1_lat, top1_lon, centroid_lat[top_indices], centroid_lon[top_indices])
	# The top-1 cell is always distance 0 from itself, so it's always kept and the
	# weights below never sum to zero.
	# Any candidate that is too far away gets its probability set to 0, so it
	# contributes nothing to the average.
	masked_probs = np.where(distance_from_top1 <= max_distance_km, top_probs, 0.0)
	# Re-scale the remaining probabilities so they add up to 1 again. For example
	# [0.62, 0.29] (after dropping a far-away 0.09) becomes [0.68, 0.32].
	weights = masked_probs / masked_probs.sum(axis=1, keepdims=True)

	# The weighted average itself: multiply each candidate's latitude/longitude by
	# its weight and add them up (the same maths as a weighted grade average).
	lat = np.sum(weights * centroid_lat[top_indices], axis=1)
	lon = np.sum(weights * centroid_lon[top_indices], axis=1)
	# Give back plain numbers for a single photo, or arrays for a batch.
	return (float(lat[0]), float(lon[0])) if single else (lat, lon)


def reverse_geocode(lat: float, lon: float) -> dict[str, str]:
	"""Return {'region', 'country_name', 'country_code'} for a coordinate via BigDataCloud.

	Best-effort: on any network error, all three values come back as empty strings
	rather than raising, since this is always used for display/analysis, never for
	something the caller can't proceed without.
	"""
	# "Reverse geocoding" is the opposite of searching a map for a place name:
	# we already have the coordinates, and we want the name of the place.
	# There is NO machine learning here - the web service simply checks which
	# country's and region's borders contain this point.
	try:
		# Send an HTTP GET request, e.g.
		#   ...reverse-geocode-client?latitude=21.03&longitude=105.85&localityLanguage=en
		# timeout=5 means: give up after 5 seconds instead of freezing the app.
		response = requests.get(
			REVERSE_GEOCODE_URL,
			params={"latitude": lat, "longitude": lon, "localityLanguage": "en"},
			timeout=5,
		)
		# Turn an HTTP error status (e.g. 404 Not Found, 429 Too Many Requests,
		# 500 Server Error) into a Python exception, handled below.
		response.raise_for_status()
		# The reply is JSON text; .json() converts it into a Python dictionary.
		data = response.json()
		# Pick out only the three fields this project uses. `.get(..., "")`
		# returns an empty string if a field is missing (for example, a point in
		# the open ocean has no country at all). "principalSubdivision" is the
		# state/province; if that's missing we fall back to the city name.
		return {
			"region": data.get("principalSubdivision") or data.get("city") or "",
			"country_name": data.get("countryName", ""),
			"country_code": data.get("countryCode", ""),
		}
	except requests.RequestException as exc:
		# Print rather than raise: this is always used for display/analysis, never
		# for something the caller can't proceed without. But a silently-swallowed
		# failure is indistinguishable from "this point legitimately has no
		# country" (e.g. open water) - print so a real API/network failure is at
		# least visible in the app's logs instead of a total guessing game.
		print(f"reverse_geocode({lat}, {lon}) failed: {exc!r}")
		return {"region": "", "country_name": "", "country_code": ""}
	except ValueError as exc:
		# The server replied, but not with valid JSON (e.g. an HTML error page).
		print(f"reverse_geocode({lat}, {lon}) got an unparseable response: {exc!r}")
		return {"region": "", "country_name": "", "country_code": ""}
