# Compliancy Detector POC
System that identifies "forbidden" or non-compliant items (e.g., weapons, drugs, explicit content) in a marketplace environment using image analysis.

System is using PyTorch to import [google/siglip2-base-patch16-512](https://huggingface.co/google/siglip2-base-patch16-512) Vision-Language encoder.


## Contents

- [Item detection flow](#item-detection-flow)
- [Project structure](#project-structure)
- [Quick Start](#quick-start)
	- [Prepare Vespa environment](#prepare-vespa-environment)
	- [Prepare Python Environment](#prepare-python-environment)
- [Python CLI (feed / detect)](#python-cli-feed--detect)


## Item detection flow

High-level flow for detecting whether an item is forbidden:

- Image input: an image file is provided to the detector (either a single image for `detect` or many images for `feed`).
- Embedding generation: the CLI uses `detector.embedding.EmbeddingGenerator` which loads the `google/siglip2-base-patch16-512` model and produces a 768-d normalized image embedding.
- Feeding: the `feed` mode creates Vespa documents with fields `item_id`, `source_path` and `image_embedding` (tensor). Documents are sent to Vespa using the Documents API.
- Indexing / Vespa schema: the Vespa schema (`compiancy-vap/schemas/item.sd`) must define a `tensor<float>(x[768])` field for `image_embedding` and ranking profiles (e.g., `closeness`) used for nearest-neighbor search.
- Querying: the `detect` mode converts the query image into a tensor literal and issues a YQL nearestNeighbor query against Vespa. Vespa returns hits with per-hit `relevance` scores.
- Decision: the CLI averages the returned relevance scores and maps the average to a human label:
	- avg > 0.80 -> "Forbidden with X% confidence."
	- 0.75 <= avg <= 0.80 -> "Needs review X% confidence."
	- avg < 0.75 -> "Safe"

This flow keeps embedding generation isolated (single model load) and relies on Vespa's nearest-neighbor ranking to retrieve visually similar items.

## Project structure

The repository contains the Vespa application definition, the Vespa document schema for items, and a place for the detector and dataset.

```
compiancy-vap/
	services.xml          - Vespa application definition (items application)
	schemas/
		item.sd             - Vespa document schema for `item`
detector/               - Detector code for item feeding and retrieval
README.md
```

Locations:

- Vespa application (services): `compiancy-vap/services.xml`
- Vespa schema for items: `compiancy-vap/schemas/item.sd`
- Items application: `detector/` 

## Quick Start

### Prepare Vespa environment

Install [Vespa CLI](https://docs.vespa.ai/en/vespa-cli.html):
<pre>
$ brew install vespa-cli
</pre>

For local deployment using docker image:
<pre data-test="exec">
$ vespa config set target local
</pre>

Pull and start the vespa docker container image:
<pre data-test="exec">
$ docker pull vespaengine/vespa
$ docker run --detach --name vespa --hostname vespa-container \
  --publish 127.0.0.1:8080:8080 --publish 127.0.0.1:19071:19071 \
  vespaengine/vespa
</pre>

Verify that configuration service (deploy api) is ready:
<pre data-test="exec">
$ vespa status deploy --wait 300
</pre>

Deploy Vespa application:
<pre data-test="exec">
$ vespa deploy ./compliancy-vap
</pre>

### Prepare Python Environment

Create a Python virtual environment and install dependencies required by the detector code. From the repository root:

```bash
# create a venv (macOS / Linux)
python3 -m venv .venv
source .venv/bin/activate

# install dependencies (uses detector/requirements.txt)
pip install -r detector/requirements.txt
```

### Python CLI (feed / detect)

A small CLI is included to generate embeddings and either feed them to Vespa or run a single-image detection query. The entrypoint is `detector/main.py`.

- Feed a folder of images to Vespa (creates documents):

```bash
python3 -m detector.main --mode feed \
	--images-folder IMG_FOLDER_PATH \
	--vespa-url http://localhost:8080 \
	--doc-type forbidden
```

- Run a detection query for a single image (nearest neighbor search):

```bash
python3 -m detector.main --mode detect \
	--image IMG_PATH \
	--vespa-url http://localhost:8080 \
	--doc-type forbidden
```

Common options:
- `--vespa-url`: Base URL of Vespa (default: `http://localhost:8080`).
- `--mode`: `feed` to upload a folder of images, `detect` to query a single image.
- `--images-folder`: Path to folder with images (required for `feed`).
- `--image`: Path to a single image file (required for `detect`).
- `--doc-type`: Document type / schema name in Vespa (default: `forbidden`).
