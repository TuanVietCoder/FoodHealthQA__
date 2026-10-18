"""Run reproducible local OCR predictions for the packaging-image collection."""

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.constraint_model import ALLERGEN_KEYWORDS
from modules.ocr_extractor import extract_with_paddle_ocr, parse_ingredients_from_ocr
from modules.pipeline import foodontorag_entity_linking, load_knowledge_base


IMAGE_DIR = os.path.join(PROJECT_ROOT, "assets", "images")
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "data", "image_ocr_predictions.csv")


def _json(value):
    return json.dumps(value, ensure_ascii=False)


def run_ocr(images_dir, output_path, limit=None, start_index=0):
    if not os.path.isdir(images_dir):
        raise NotADirectoryError(f"Image directory not found: {images_dir}")
    image_paths = sorted(
        name for name in os.listdir(images_dir)
        if name.lower().endswith((".jpeg", ".jpg", ".png", ".webp"))
    )
    image_rows = [
        {
            "image_id": os.path.splitext(name)[0],
            "image_path": os.path.relpath(
                os.path.join(images_dir, name), PROJECT_ROOT
            ),
        }
        for name in image_paths
    ]
    if not image_rows:
        raise ValueError(f"No supported image files found in {images_dir}")

    selected_rows = image_rows[start_index:]
    if limit is not None:
        selected_rows = selected_rows[:limit]
    if not selected_rows:
        raise ValueError("No images selected; check start-index and limit")

    knowledge_base = load_knowledge_base(os.path.join(PROJECT_ROOT, "data"))
    output_fields = [
        "image_id",
        "image_path",
        "raw_ocr_text",
        "normalized_text",
        "raw_ingredients_text",
        "normalized_ingredients_text",
        "normalized_ingredients_json",
        "additive_codes_json",
        "allergens_json",
        "ocr_confidence",
        "ocr_engine",
        "gemini_spell_correction_used",
        "ocr_error",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    file_exists = os.path.exists(output_path) and start_index > 0
    mode = "a" if file_exists else "w"
    with open(output_path, mode, encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=output_fields)
        if mode == "w":
            writer.writeheader()

        for row in selected_rows:
            image_path = os.path.join(PROJECT_ROOT, row["image_path"])
            result = extract_with_paddle_ocr(
                image_path,
                sample_key="custom",
                use_gemini_correction=False,
            )
            mentions = result.get("mentions", [])
            raw_mentions = parse_ingredients_from_ocr(
                result.get("raw_text_before_norm", "")
            )
            linked = [
                foodontorag_entity_linking(mention, knowledge_base)
                for mention in mentions
            ]
            normalized_ingredients = []
            additive_codes = []
            for item in linked:
                entity = item.get("matched_entity")
                normalized_ingredients.append(
                    entity.get("canonical_name")
                    if entity
                    else item.get("raw_mention", "")
                )
                if entity and entity.get("ins_code"):
                    additive_codes.append(entity["ins_code"])

            ingredient_text = "; ".join(
                mention.get("raw_mention", "") for mention in mentions
            )
            allergen_text = f"{result.get('full_text', '')} {ingredient_text}".casefold()
            allergens = sorted({
                allergen
                for allergen, keywords in ALLERGEN_KEYWORDS.items()
                if any(keyword.casefold() in allergen_text for keyword in keywords)
            })
            writer.writerow({
                "image_id": row["image_id"],
                "image_path": row["image_path"],
                "raw_ocr_text": result.get("raw_text_before_norm", ""),
                "normalized_text": result.get("full_text", ""),
                "raw_ingredients_text": "; ".join(
                    mention.get("raw_mention", "") for mention in raw_mentions
                ),
                "normalized_ingredients_text": ingredient_text,
                "normalized_ingredients_json": _json(normalized_ingredients),
                "additive_codes_json": _json(sorted(set(additive_codes))),
                "allergens_json": _json(allergens),
                "ocr_confidence": result.get("avg_confidence", 0.0),
                "ocr_engine": result.get("engine_name", ""),
                "gemini_spell_correction_used": result.get("llm_spell_used", False),
                "ocr_error": result.get("ocr_error", ""),
            })
            target.flush()
            print(f"{row['image_id']}: {'error' if result.get('is_ocr_unavailable') else 'ok'}")
    print(f"Predictions written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", default=IMAGE_DIR)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start-index", type=int, default=0)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer")
    if args.start_index < 0:
        parser.error("--start-index must be non-negative")
    run_ocr(args.images_dir, args.output, args.limit, args.start_index)


if __name__ == "__main__":
    main()
