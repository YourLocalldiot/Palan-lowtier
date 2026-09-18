"""Geospatial helpers shared by the project scripts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


EARTH_RADIUS_KM = 6371.0088

# Standard geolocation error buckets (street / city / region / country / continent).
ACCURACY_THRESHOLDS_KM = (1.0, 25.0, 200.0, 750.0, 2500.0)


def haversine(
	lat1: float | np.ndarray,
	lon1: float | np.ndarray,
	lat2: float | np.ndarray,
	lon2: float | np.ndarray,
) -> float | np.ndarray:
	"""Return great-circle distance in kilometres between coordinate pairs."""
	lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(
		np.radians, (lat1, lon1, lat2, lon2)
	)
	delta_lat = lat2_rad - lat1_rad
	delta_lon = lon2_rad - lon1_rad
	haversine_angle = (
		np.sin(delta_lat / 2.0) ** 2
		+ np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2.0) ** 2
	)
	distance = 2.0 * EARTH_RADIUS_KM * np.arcsin(
		np.sqrt(np.clip(haversine_angle, 0.0, 1.0))
	)
	return float(distance) if np.ndim(distance) == 0 else distance


def load_centroids(path: Path) -> dict[int, tuple[float, float]]:
	"""Load a cell_centroids.json mapping cell id to (lat, lon)."""
	data = json.loads(Path(path).read_text(encoding="utf-8"))
	return {int(cell_id): (value["centroid_lat"], value["centroid_lon"]) for cell_id, value in data.items()}


def accuracy_at_thresholds(
	distances_km: np.ndarray, thresholds_km: tuple[float, ...] = ACCURACY_THRESHOLDS_KM
) -> dict[float, float]:
	"""Return the fraction of distances at or under each threshold, in kilometres."""
	distances_km = np.asarray(distances_km, dtype=np.float64)
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
	probabilities = np.asarray(probabilities, dtype=np.float64)
	single = probabilities.ndim == 1
	if single:
		probabilities = probabilities[np.newaxis, :]

	k = min(top_k, probabilities.shape[1])
	top_indices = np.argsort(probabilities, axis=1)[:, ::-1][:, :k]
	top_probs = np.take_along_axis(probabilities, top_indices, axis=1)

	top1_lat = centroid_lat[top_indices[:, :1]]
	top1_lon = centroid_lon[top_indices[:, :1]]
	distance_from_top1 = haversine(top1_lat, top1_lon, centroid_lat[top_indices], centroid_lon[top_indices])
	# The top-1 cell is always distance 0 from itself, so it's always kept and the
	# weights below never sum to zero.
	masked_probs = np.where(distance_from_top1 <= max_distance_km, top_probs, 0.0)
	weights = masked_probs / masked_probs.sum(axis=1, keepdims=True)

	lat = np.sum(weights * centroid_lat[top_indices], axis=1)
	lon = np.sum(weights * centroid_lon[top_indices], axis=1)
	return (float(lat[0]), float(lon[0])) if single else (lat, lon)
