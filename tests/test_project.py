import unittest, tempfile, json
from pathlib import Path
from project_demo.runtime import parse_object, wilson


class SharedTests(unittest.TestCase):
    def test_no_letter_mining(self):
        with self.assertRaises(ValueError):
            parse_object("An answer cannot be determined")
        self.assertEqual(parse_object('Analysis text. {"answer":"D"}')["answer"], "D")

    def test_confidence_interval(self):
        lo, hi = wilson(0, 3)
        self.assertLessEqual(lo, 1e-10)
        self.assertGreater(hi, 0.4)
        self.assertEqual(wilson(0, 0), [None, None])


from project_demo import core
from project_demo.tokenization import FixtureTokenizer, pad_exact
from project_demo.designer import validate_template


class ReasoningTests(unittest.TestCase):
    def test_oracles(self):
        self.assertEqual(
            core.oracle(
                "pointer", {"start": "a", "steps": 2, "links": {"a": "b", "b": "c"}}
            ),
            "c",
        )
        self.assertEqual(
            core.oracle(
                "arithmetic", {"start": 96, "operations": [("add", 3), ("multiply", 4)]}
            ),
            "8",
        )
        self.assertEqual(core.oracle("brackets", {"sequence": "([)]"}), "invalid")
        self.assertEqual(core.oracle("brackets", {"sequence": "([])"}), "valid")

    def test_nesting_and_display_length_independent(self):
        for depth in [1, 3, 10]:
            p, _ = core.make_problem("brackets", depth, 42)
            self.assertEqual(len(p["sequence"]), 40)
            self.assertEqual(core.oracle("brackets", p), "valid")

    def test_paired_conditions_preserve_task(self):
        c = dict(core.DEFAULT_CONFIG, replicates=1, difficulties=[1, 3])
        rows = core.prepare(c, "fixture", Path("."))
        self.assertEqual(len(rows), 12)
        for i in range(0, len(rows), 2):
            self.assertEqual(rows[i]["payload"], rows[i + 1]["payload"])
            self.assertEqual(rows[i]["answer"], rows[i + 1]["answer"])
            self.assertEqual(len(rows[i]["prompt"]), rows[i]["length"])

    def test_fixture_is_oracle_not_model(self):
        c = dict(core.DEFAULT_CONFIG, replicates=1, difficulties=[1], lengths=[1024])
        rows = core.prepare(c, "fixture", Path("."))
        result = core.run(c, rows, "fixture", Path("."))
        self.assertTrue(all(r["correct"] and r["mode"] == "fixture" for r in result))

    def test_template_integrity(self):
        self.assertTrue(
            validate_template("Please solve {TASK} and return JSON.", "paraphrase")
        )
        self.assertFalse(validate_template("Use {ANSWER} for {TASK}.", "paraphrase"))

    def test_bad_budget_rejected(self):
        with self.assertRaises(ValueError):
            pad_exact("too long", 2, FixtureTokenizer())
