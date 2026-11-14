import csv
import json
import unittest
from pathlib import Path

from modules.additive_linker import (
    link_additive_mentions,
    link_additive_name,
    load_additive_registry,
)
from modules.benchmark_manifest import (
    get_active_benchmark_path,
    is_gold_eligible,
    load_benchmark_manifest,
)
from modules.hybrid_retriever import foodontorag_4agent_linking
from modules.pipeline import evaluate_safety_and_guardrail


ROOT = Path(__file__).resolve().parents[1]


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class AdditiveRegistryTests(unittest.TestCase):
    def test_registry_contains_400_unique_tt24_identities(self):
        registry = load_additive_registry()
        self.assertEqual(len(registry), 400)
        self.assertEqual(len({item["ins_key"] for item in registry}), 400)
        self.assertTrue(all(item.get("identity_only") for item in registry))

    def test_exact_names_map_to_expected_ins_codes(self):
        cases = {
            "dl-alpha Tocopherol": "INS 307c",
            "Chất nhũ hóa: Lecithin đậu nành": "INS 322(i)",
            "Caramen nhóm IV (caramen amoni sulfit)": "INS 150d",
            "Caramen nhóm I (caramen nguyên chất)": "INS 150a",
        }
        for mention, expected in cases.items():
            with self.subTest(mention=mention):
                match = link_additive_name(mention)
                self.assertIsNotNone(match)
                self.assertEqual(match["ins_code"], expected)
                self.assertEqual(match["code_origin"], "inferred_from_name")

    def test_ambiguous_family_names_are_not_auto_linked(self):
        self.assertEqual(link_additive_mentions("Tocopherol"), [])
        self.assertEqual(link_additive_mentions("Caramen"), [])
        self.assertEqual(link_additive_mentions("Hương vanilla tổng hợp"), [])

    def test_bare_code_requires_additive_context(self):
        linked = link_additive_mentions("chất ổn định [170(i)]")
        self.assertEqual([item["ins_code"] for item in linked], ["INS 170(i)"])
        self.assertEqual(link_additive_mentions("khối lượng 170 g"), [])

    def test_hybrid_pipeline_prefers_deterministic_registry(self):
        result = foodontorag_4agent_linking(
            {
                "raw_mention": "Chất tạo màu: Caramen nhóm IV",
                "category": "color",
            },
            [],
        )
        self.assertEqual(result["matched_entity"]["ins_code"], "INS 150d")
        self.assertEqual(result["code_origin"], "inferred_from_name")
        self.assertEqual(result["retrieval_engine"], "Deterministic additive identity registry")

    def test_unlinked_food_ingredients_do_not_lower_additive_kb_coverage(self):
        linked = []
        for mention in (
            "dl-alpha Tocopherol",
            "Lecithin đậu nành",
            "Caramen nhóm IV",
            "Caramen nhóm I",
        ):
            match = link_additive_name(mention)
            linked.append(
                {
                    "raw_mention": mention,
                    "category": "additive",
                    "matched_entity": match["matched_entity"],
                    "retrieval_confidence": 0.99,
                }
            )
        linked.extend(
            {
                "raw_mention": ingredient,
                "category": "ingredient",
                "matched_entity": None,
                "retrieval_confidence": 0.0,
            }
            for ingredient in ("bột mì", "đường", "whey", "tinh bột ngô", "cà phê")
        )

        identity = evaluate_safety_and_guardrail(
            linked,
            {"conditions": [], "allergies": [], "weight_kg": 60},
            "Sản phẩm có phụ gia nào?",
            avg_ocr_conf=1.0,
        )
        self.assertEqual(identity["status"], "SCREENED_NO_ALERT")

        exposure = evaluate_safety_and_guardrail(
            linked,
            {"conditions": [], "allergies": [], "weight_kg": 60},
            "Ăn hằng ngày có an toàn không?",
            avg_ocr_conf=1.0,
        )
        self.assertEqual(exposure["status"], "ABSTAIN")
        self.assertNotIn("chụp lại", " ".join(exposure["abstention_reasons"]).casefold())


def get_active_benchmark_file(root):
    return Path(get_active_benchmark_path(root / "data", "jsonl"))


class ActiveBenchmarkTests(unittest.TestCase):
    def test_manifest_selects_only_active_version(self):
        manifest = load_benchmark_manifest(ROOT / "data")
        active = Path(get_active_benchmark_path(ROOT / "data", "jsonl"))
        self.assertEqual(manifest["active_version"], "v5")
        self.assertEqual(active.name, Path(manifest["benchmarks"]["v5"]["jsonl"]).name)
        self.assertTrue(active.exists())
        # one standard dataset: only v5 is registered
        self.assertEqual(list(manifest["benchmarks"]), ["v5"])
        self.assertEqual(active.name, "qa_benchmark.jsonl")

    def test_active_benchmark_is_never_eligible_as_gold(self):
        manifest = load_benchmark_manifest(ROOT / "data")
        metadata = manifest["benchmarks"][manifest["active_version"]]
        self.assertFalse(metadata["expert_validated"])
        with get_active_benchmark_file(ROOT).open(encoding="utf-8") as source:
            active_rows = [json.loads(line) for line in source if line.strip()]
        self.assertTrue(all(not is_gold_eligible(row, metadata) for row in active_rows))


if __name__ == "__main__":
    unittest.main()
