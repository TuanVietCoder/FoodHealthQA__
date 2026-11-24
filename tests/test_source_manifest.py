"""Manifest nguồn phải khớp các file thực sự có trong knowledge_base/sources/raw/external (sha256, số byte, không sót file)."""
import hashlib
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts_research"))
import sync_jecfa_manifest as sync  # noqa: E402

EXTERNAL = ROOT / "knowledge_base" / "sources" / "raw" / "external"
MANIFEST = json.loads(sync.MANIFEST.read_text(encoding="utf-8"))


class JecfaManifestTests(unittest.TestCase):
    def test_every_jecfa_page_on_disk_is_registered_and_matches_its_hash(self):
        self.assertEqual(sync.audit(MANIFEST), {})

    def test_audit_detects_each_kind_of_drift(self):
        pages = MANIFEST["jecfa_pages"]
        first = pages[0]
        tampered = {"jecfa_pages": [dict(first, sha256="0" * 64)] + pages[1:]}
        self.assertEqual(sync.audit(tampered), {"sha256 lệch": [first["file"].split("/")[-1]]})
        shorter = {"jecfa_pages": pages[1:]}
        self.assertEqual(sync.audit(shorter), {"chưa đăng ký": [first["file"].split("/")[-1]]})
        ghost = {"jecfa_pages": pages + [dict(first, file="jecfa/Chemical_999999.html", source_id="JECFA_CHEMICAL_999999")]}
        self.assertEqual(sync.audit(ghost), {"mất file": ["Chemical_999999.html"]})

    def test_registered_urls_and_ids_are_consistent(self):
        for e in MANIFEST["jecfa_pages"]:
            cid = e["source_id"].rsplit("_", 1)[1]
            self.assertTrue(e["url"].endswith(f"/Home/Chemical/{cid}"), e["source_id"])
            self.assertEqual(e["file"], f"jecfa/Chemical_{cid}.html")


class ManuallyDownloadedManifestTests(unittest.TestCase):
    def test_manually_downloaded_files_that_exist_match_their_hash(self):
        checked = 0
        for e in MANIFEST["manually_downloaded"]:
            path = EXTERNAL / e["file"]
            if not path.is_file():
                continue  # tệp có thể được đặt ở thư mục khác (vn/); chỉ kiểm tệp có mặt
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), e["sha256"], e["file"])
            checked += 1
        self.assertGreater(checked, 20)


if __name__ == "__main__":
    unittest.main()
