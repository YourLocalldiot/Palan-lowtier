# Palan Lowtier

Palan Lowtier is a machine learning project for working with streamed geographic and language data across Vietnam, the Philippines, Japan, and South Korea.

## Current Scope

The current project scope covers:

- VN: Vietnam
- PH: Philippines
- JP: Japan
- KR: South Korea

## Setup

Create and activate a virtual environment, then install the pinned dependencies:

```bash
pip install -r requirements.txt
```

## Run Order

Run the project stages in this order:

```text
python data_pipeline_1.py -> python train_2.py -> python evaluate_3.py -> python export_4.py -> streamlit run app.py
```

The `data/` and `checkpoints/` directories are regenerated locally and are not stored in git; they contain generated dataset files and model checkpoints. The `export/` directory *is* committed — it's the small TFLite bundle that the Streamlit app (including the deployed Streamlit Cloud app) reads directly, so it needs to be in the repo.

## Data Pipeline

Run the pipeline after installing the requirements:

```bash
python data_pipeline_1.py
```

It streams and filters the training split, caches images in
`data/raw/filtered_train.tfrecord`, assigns 100 KMeans geocells, writes
`cell_centroids.json`, and stores labeled examples in
`data/processed/filtered_train_with_cells.tfrecord`. The pipeline also prints
country counts and the global-centroid haversine baseline.

For an internal split by `panoid` (never by individual image), use for example:

```bash
python data_pipeline_1.py --dev-fraction 0.2 --normalization imagenet
```
