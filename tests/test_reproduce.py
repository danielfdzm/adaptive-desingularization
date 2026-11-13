"""Integrity and isolation checks for the release interface; no scientific runs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BUILDERS = ("build_pb_solution_maps", "build_algorithm", "build_forcing_solutions",
            "build_local", "build_recovery_tradeoff", "build_budget", "build_supplement")


class ReproductionInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        shutil.copy2(ROOT / "reproduce.py", self.root / "reproduce.py")
        workspace = self.root / "reproduction/workspace"
        records = workspace / "example"
        records.mkdir(parents=True)
        (records / "record.json").write_text('{"value": 7}\n')
        (records / "analyze.py").write_text(
            "from pathlib import Path\n"
            "p = Path(__file__).parent\n"
            "(p / 'record.json').write_text('changed by builder\\n')\n"
            "(p / 'table.csv').write_text('answer\\n7\\n')\n"
            "print('completed fixture analysis')\n")
        figures = workspace / "paper_1111/revision-validation/publication-figures"
        figures.mkdir(parents=True)
        for name in BUILDERS:
            (figures / (name + ".py")).write_text("# unused fixture builder\n")
        (self.root / "experiments").mkdir()
        (self.root / "experiments/catalog.json").write_text(json.dumps({
            "schema_version": 1, "path_base": "reproduction/workspace",
            "experiments": [{"id": "example", "title": "Example", "records": ["example"],
                             "analysis": [{"script": "example/analyze.py", "kind": "saved_records"}],
                             "rerun": []}],
        }))
        self.write_manifest()

    def write_manifest(self):
        lines = [f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(self.root).as_posix()}\n"
                 for path in sorted(self.root.rglob("*"))
                 if path.is_file() and path.name != "SHA256SUMS"]
        (self.root / "SHA256SUMS").write_text("".join(lines))

    def run_cli(self, *arguments):
        return subprocess.run([sys.executable, str(self.root / "reproduce.py"), *arguments],
                              cwd=self.temporary.name, capture_output=True, text=True)

    def test_valid_inventory_and_ignored_generated_outputs(self):
        (self.root / "outputs").mkdir()
        (self.root / "outputs/generated.csv").write_text("unversioned output\n")
        result = self.run_cli("check")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_missing_altered_and_unlisted_records_fail(self):
        record = self.root / "reproduction/workspace/example/record.json"
        original = record.read_bytes()
        record.unlink()
        self.assertIn("Missing files", self.run_cli("check").stderr)
        record.write_text("altered\n")
        self.assertIn("Changed files", self.run_cli("check").stderr)
        record.write_bytes(original)
        record.with_name("unreported.json").write_text("{}")
        self.assertIn("Unlisted files", self.run_cli("check").stderr)

    def test_catalog_paths_cannot_escape_package(self):
        path = self.root / "experiments/catalog.json"
        catalog = json.loads(path.read_text())
        catalog["experiments"][0]["records"] = ["../../outside"]
        path.write_text(json.dumps(catalog))
        self.write_manifest()
        result = self.run_cli("check")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("inside the release", result.stderr)

    def test_table_builder_writes_only_to_disposable_copy(self):
        record = self.root / "reproduction/workspace/example/record.json"
        before = record.read_bytes()
        result = self.run_cli("tables", "example")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(record.read_bytes(), before)
        output = self.root / "outputs/tables/workspace/example/table.csv"
        self.assertEqual(output.read_text(), "answer\n7\n")
        self.assertEqual(self.run_cli("check").returncode, 0)
        self.assertFalse(list((self.root / "outputs").glob(".reproduce-*")))

    def test_manuscripts_rejected_even_after_checksums_are_updated(self):
        cases = {
            "paper.tex": (r"\section{Manuscript}", "Manuscript sources"),
            "draft.pdf": ("%PDF-1.7", "Only catalogued figure PDFs"),
            "table.tex": (r"\documentclass{article}", "Only numerical table fragments"),
            "bundle.zip": ("archive", "not bundled archives"),
        }
        for name, (content, message) in cases.items():
            with self.subTest(name=name):
                path = self.root / name
                path.write_text(content)
                self.write_manifest()
                result = self.run_cli("check")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)
                path.unlink()


if __name__ == "__main__":
    unittest.main()
