"""Tests for dynamic claim locator and source resolver."""

import sys
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.claim_locator_resolver import (
    resolve_dynamic_locators,
    resolve_source_urls,
    extract_ins_keys,
    STATUTORY_SOURCE_CATALOG,
)


class TestClaimLocatorResolver(unittest.TestCase):
    def test_ckd_phosphate_efsa_resolution(self):
        row = {
            "id": "TEST_001",
            "domain": "vulnerable_groups",
            "persona_id": "ckd_stage3",
            "question": "Tôi bị suy thận mạn giai đoạn 3 — xét về nhóm phụ gia chứa gốc photphat vô cơ trong 'Bánh dừa nướng'...",
            "ins_codes_matched": "INS322(I), INS450(I), INS500(II)",
            "source_citations": "EFSA_OPINIONS, CODEX_STAN_192_1995",
        }
        locators = resolve_dynamic_locators(row)
        self.assertIn("EFSA_ANS_2019_5674#Phosphates_ADI40", locators)
        self.assertNotIn("Glutamates", locators)
        self.assertIn("CODEX_STAN_192_1995#Table3_GMP", locators)

    def test_child_hyperactivity_azo_dyes_efsa_resolution(self):
        row = {
            "id": "TEST_002",
            "domain": "vulnerable_groups",
            "persona_id": "child_5yo",
            "question": "Xét riêng về phụ gia và phẩm màu trong bánh, sản phẩm có an toàn cho trẻ nhỏ 5 tuổi trước nguy cơ tăng động không?",
            "ins_codes_matched": "INS102, INS110, INS471",
            "source_citations": "EFSA_OPINIONS",
        }
        locators = resolve_dynamic_locators(row)
        self.assertIn("EFSA_ANS_2008_852#AzoDyes_Hyperactivity_Children", locators)
        self.assertNotIn("Glutamates", locators)

    def test_pregnancy_nitrites_efsa_resolution(self):
        row = {
            "id": "TEST_003",
            "domain": "vulnerable_groups",
            "persona_id": "pregnant_woman",
            "question": "Thai phụ có thể sử dụng xúc xích (chứa INS250, INS320) không hay cần thận trọng?",
            "ins_codes_matched": "INS250, INS320",
            "source_citations": "EFSA_OPINIONS, CODEX_STAN_192_1995",
        }
        locators = resolve_dynamic_locators(row)
        self.assertIn("EFSA_ANS_2017_4786#Nitrite_Methemoglobin", locators)
        self.assertNotIn("Glutamates", locators)

    def test_jecfa_non_aspartame_product_resolution(self):
        row = {
            "id": "TEST_004",
            "domain": "insufficient_data",
            "persona_id": "healthy_adult",
            "question": "Tôi uống Tea Break mỗi ngày, lượng phụ gia INS300, INS471, INS500(II), INS950, INS955 có bị tích lũy vượt ngưỡng ADI không?",
            "ins_codes_matched": "INS300, INS471, INS500(II), INS950, INS955",
            "source_citations": "JECFA_EVALUATIONS, CODEX_STAN_192_1995",
        }
        locators = resolve_dynamic_locators(row)
        self.assertNotIn("Chemical62_Aspartame", locators)
        self.assertIn("Chemical1845_AscorbicAcid_GMP", locators)
        self.assertIn("CODEX_STAN_192_1995#Table3_GMP", locators)

    def test_jecfa_aspartame_product_resolution(self):
        row = {
            "id": "TEST_005",
            "domain": "insufficient_data",
            "persona_id": "healthy_adult",
            "question": "Tôi uống trà mỗi ngày, lượng phụ gia INS150D, INS951 có bị tích lũy vượt ngưỡng ADI không?",
            "ins_codes_matched": "INS150D, INS951",
            "source_citations": "JECFA_EVALUATIONS, CODEX_STAN_192_1995",
        }
        locators = resolve_dynamic_locators(row)
        self.assertIn("WHO_FAO_JECFA_2023#Chemical62_Aspartame", locators)

    def test_pku_falcpa_resolution(self):
        row = {
            "id": "TEST_006",
            "domain": "vulnerable_groups",
            "persona_id": "pku_patient",
            "question": "Tôi mắc bệnh PKU — chất tạo ngọt aspartame có an toàn không?",
            "ins_codes_matched": "INS951",
            "source_citations": "US_FDA_FALCPA, CODEX_STAN_192_1995",
        }
        locators = resolve_dynamic_locators(row)
        self.assertIn("US_FDA_21CFR172.804#b", locators)
        self.assertNotIn("US_FDA_FALCPA_2004#Sec203", locators)

    def test_normal_allergen_falcpa_resolution(self):
        row = {
            "id": "TEST_007",
            "domain": "allergen_detection",
            "persona_id": "peanut_allergy",
            "question": "Sản phẩm có chứa đậu phộng không?",
            "ins_codes_matched": "",
            "source_citations": "US_FDA_FALCPA, WHO_JEMRA_PRIORITY_ALLERGENS",
        }
        locators = resolve_dynamic_locators(row)
        self.assertIn("US_FDA_FALCPA_2004#Sec203", locators)
        self.assertIn("FAO_WHO_JEMRA_2021_PART1#PriorityAllergens", locators)

    def test_url_resolution(self):
        locators = "EFSA_ANS_2019_5674#Phosphates_ADI40; VN_TT24_2019#PhuLuc1_Bang1; WHO_FAO_JECFA_2023#Chemical62_Aspartame"
        urls = resolve_source_urls(locators)
        self.assertEqual(len(urls), 3)
        self.assertTrue(any("chinhphu.vn" in u for u in urls))
        self.assertTrue(any("efsa" in u for u in urls))
        self.assertTrue(any("who.int" in u for u in urls))


if __name__ == "__main__":
    unittest.main()
