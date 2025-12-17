import argparse
import hashlib
import sys
from pathlib import Path

import requests

from detector.embedding import EmbeddingGenerator


def stable_id_for_path(p: str) -> str:
    h = hashlib.sha1(p.encode("utf-8")).hexdigest()
    return h


def vespa_tensor_768(values: list[float]) -> str:
    stingified_values = ",".join(f"{v:.7f}" for v in values)
    return "tensor<float>(x[768]):[" + stingified_values + "]"


def feed_images(vespa_url: str,
                namespace: str,
                doc_type: str,
                images_path: str):
    gen = EmbeddingGenerator()
    folder = Path(images_path)
    filenames = [p.name for p in folder.iterdir() if p.is_file()]

    for filename in filenames:
        img_path = images_path + "/" + filename
        prepared_embeddings = gen.embed_from_path(img_path)
        docid = stable_id_for_path(img_path)

        fields = {
            "item_id": docid,
            "source_path": img_path,
            "image_embedding": prepared_embeddings,
        }

        endpoint = f"{vespa_url}/document/v1/{namespace}/{doc_type}/docid/{docid}"
        payload = {"fields": fields}
        r = requests.post(endpoint, json=payload, timeout=30)
        if r.ok:
            print(f"OK  {img_path}  -> docid={docid}")
        else:
            print(
                f"FAIL {img_path} -> {r.status_code} {r.text}",
                file=sys.stderr
            )


def detect_image(vespa_url: str,
                 doc_type: str,
                 image_path: str,
                 hits: int = 3):
    gen = EmbeddingGenerator()
    prepared_embeddings = gen.embed_from_path(image_path)

    body = {
        "yql": f"select item_id, source_path from {doc_type} where ({{targetHits: 10}}nearestNeighbor(image_embedding, q));",
        "hits": hits,
        "ranking.profile": "closeness",
        "input.query(q)": vespa_tensor_768(prepared_embeddings),
    }

    resp = requests.post(f"{vespa_url.rstrip('/')}/search/", json=body, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    
    children = data.get("root", {}).get("children", [])
    if not children:
        print("No hits returned")
        return 0.0

    avg = sum(float(match.get("relevance", 0.0)) for match in children) / len(children)

    pct = avg * 100.0
    if avg > 0.8:
        result = f"Forbidden with {pct:.1f}% confidence."
    elif avg >= 0.75:
        result = f"Needs review {pct:.1f}% confidence."
    else:
        result = f"Safe {pct:.1f}%."

    print(result)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Vespa image detector/feeder CLI")
    parser.add_argument("--vespa-url", default="http://localhost:8080", help="Vespa base URL")
    parser.add_argument("--mode", required=True, choices=["detect", "feed"], help="Operation mode")
    parser.add_argument("--images-folder", help="Folder with images to feed (feed mode)")
    parser.add_argument("--image", help="Single image path to detect (detect mode)")
    parser.add_argument("--doc-type", default="forbidden", help="Document type/schema name in Vespa")

    args = parser.parse_args(argv)

    namespace = "vinted"

    if args.mode == "feed":
        if not args.images_folder:
            parser.error("--images-folder is required when mode=feed")
        feed_images(args.vespa_url, namespace, args.doc_type, args.images_folder)
    else:
        if not args.image:
            parser.error("--image is required when mode=detect")
        detect_image(args.vespa_url, args.doc_type, args.image)


if __name__ == "__main__":
    main()
