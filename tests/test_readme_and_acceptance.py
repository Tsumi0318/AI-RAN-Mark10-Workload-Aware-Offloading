from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_readme_contains_performance_sections_in_order() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    headings = ["### V-A.", "### V-B.", "### V-C.", "### V-D.", "### V-E.", "### V-F.", "### V-G.", "### V-H."]
    positions = [text.index(heading) for heading in headings]
    assert positions == sorted(positions)


def test_readme_has_no_embedded_images_and_covers_six_sections() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "![" not in text
    for heading in ["# I.", "# II.", "# III.", "# IV.", "# V.", "# VI."]:
        assert heading in text


def test_acceptance_audit_is_all_pass_and_preserves_api_privacy() -> None:
    audit = json.loads((ROOT / "06_审计与复现" / "acceptance_audit.json").read_text(encoding="utf-8"))
    assert audit["overall_pass"]
    assert all(row["status"] == "PASS" for row in audit["checks"])
    assert "sk-" not in json.dumps(audit).lower()
    assert any(row["requirement"] == "api_key_not_saved" for row in audit["checks"])
