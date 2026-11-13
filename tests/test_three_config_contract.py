import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from modules.pipeline import (
    build_persona_from_benchmark_record,
    extract_safety_label,
)
from scripts_research.generate_predictions import _predict_one
from scripts_research.run_evaluation import _prediction_integrity


def test_safety_label_requires_explicit_output_line():
    assert extract_safety_label("Sản phẩm có vẻ SAFE trong mô tả.") == "ABSTAIN"
    assert extract_safety_label("Kết luận\nSAFETY_LABEL: WARNING") == "WARNING"


def test_persona_mapping_uses_exact_benchmark_id():
    personas = [
        {"id": "USER_CLINICAL_001", "name": "PKU", "conditions": ["pku"], "allergies": []},
        {"id": "USER_CLINICAL_007", "name": "Healthy", "conditions": ["healthy"], "allergies": []},
    ]
    healthy = build_persona_from_benchmark_record(
        {"persona_id": "healthy_adult", "persona_name": "Người khỏe mạnh"}, personas
    )
    pku = build_persona_from_benchmark_record(
        {"persona_id": "pku_patient", "persona_name": "Bệnh nhân PKU"}, personas
    )
    assert healthy["id"] == "USER_CLINICAL_007"
    assert pku["id"] == "USER_CLINICAL_001"
    assert "phenylketonuria_pku" in pku["conditions"]

    allergy = build_persona_from_benchmark_record(
        {
            "persona_id": "allergy_patient",
            "persona_conditions": "milk_allergy",
            "question": "Các thành phần này có phù hợp không?",
        },
        personas,
    )
    assert allergy["allergies"] == ["milk"]


def test_b_and_c_call_the_same_grounded_downstream(tmp_path):
    record = {
        "id": "Q1",
        "persona_id": "healthy_adult",
        "question": "Có phù hợp không?",
        "product_name": "Sản phẩm",
        "product_ingredients": "nước, INS 330",
        "image_ref": "images/0001.jpeg",
    }
    resources = {
        "personas": [],
        "kb_additives": [],
        "hazards": [],
        "fndds": [],
        "off_products": [],
    }
    downstream_result = {
        "llm_answer": "Trả lời\nSAFETY_LABEL: SAFE",
        "safety_label": "SAFE",
        "evaluation": {"status": "SCREENED_NO_ALERT", "s_retrieval": 0.8, "s_total": 0.9, "citations": []},
        "citation_validation": {"valid": True},
    }

    with patch(
        "scripts_research.generate_predictions.run_grounded_pipeline",
        return_value=downstream_result,
    ) as downstream:
        prediction_b = _predict_one("B", record, resources, {}, tmp_path / "ocr.json")
        b_call = downstream.call_args.kwargs

    fake_ocr = {
        "full_text": "Thành phần: nước, INS 330",
        "mentions": b_call["mentions"],
        "avg_confidence": 0.73,
        "engine_name": "PaddleOCR + VietOCR",
    }
    with patch(
        "scripts_research.generate_predictions._ocr_input",
        return_value=(fake_ocr, None),
    ), patch(
        "scripts_research.generate_predictions.run_grounded_pipeline",
        return_value=downstream_result,
    ) as downstream:
        prediction_c = _predict_one("C", record, resources, {}, tmp_path / "ocr.json")
        c_call = downstream.call_args.kwargs

    assert b_call.keys() == c_call.keys()
    assert b_call["avg_ocr_conf"] == 1.0
    assert c_call["avg_ocr_conf"] == 0.73
    assert prediction_b["input_source"] == "ground_truth_text"
    assert prediction_c["input_source"] == "image_ocr"
    assert prediction_b["guardrail_status"] == prediction_c["guardrail_status"]


def test_official_evaluation_rejects_partial_or_non_real_predictions():
    references = [{"id": "Q1"}, {"id": "Q2"}]
    bad = [{"id": "Q1", "answer": "x", "evaluation_type": "legacy"}]
    integrity = _prediction_integrity(references, bad)
    assert not integrity["complete"]
    assert integrity["missing"] == ["Q2"]
    assert integrity["non_real"] == ["Q1"]


class ThreeConfigContractTests(unittest.TestCase):
    def test_safety_label_contract(self):
        test_safety_label_requires_explicit_output_line()

    def test_exact_persona_contract(self):
        test_persona_mapping_uses_exact_benchmark_id()

    def test_shared_downstream_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            test_b_and_c_call_the_same_grounded_downstream(Path(directory))

    def test_evaluation_integrity_contract(self):
        test_official_evaluation_rejects_partial_or_non_real_predictions()


if __name__ == "__main__":
    unittest.main()
