"""Tests for the local Flask demo server."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import torch

from model.config import GPTConfig
from model.gpt import GPT
from server.app import (
    MAX_NEW_TOKENS_LIMIT,
    create_app,
    load_evaluation_summary,
    resolve_generation_params,
)


def _write_checkpoint(path: Path, config: GPTConfig, step: int) -> None:
    torch.manual_seed(0)
    model = GPT(config)
    model.save_checkpoint(str(path), step=step)


class TestResolveGenerationParams(unittest.TestCase):
    """Tests for server-side validation and clamping of generation params."""

    def test_defaults_applied_when_only_prompt_given(self) -> None:
        params = resolve_generation_params({"prompt": "hello"})
        self.assertEqual(params["prompt"], "hello")
        self.assertEqual(params["max_new_tokens"], 200)
        self.assertEqual(params["temperature"], 0.8)
        self.assertEqual(params["top_k"], 50)
        self.assertEqual(params["top_p"], 0.9)
        self.assertFalse(params["greedy"])

    def test_clamps_values_above_range(self) -> None:
        params = resolve_generation_params(
            {"prompt": "p", "max_tokens": 10_000, "temperature": 99.0, "top_k": 900, "top_p": 4.0},
            vocab_size=100,
        )
        self.assertEqual(params["max_new_tokens"], MAX_NEW_TOKENS_LIMIT)
        self.assertEqual(params["temperature"], 2.0)
        self.assertEqual(params["top_k"], 100)
        self.assertEqual(params["top_p"], 1.0)

    def test_clamps_values_below_range(self) -> None:
        params = resolve_generation_params(
            {"prompt": "p", "max_tokens": -5, "temperature": -1.0, "top_k": -3, "top_p": 0.0}
        )
        self.assertEqual(params["max_new_tokens"], 1)
        self.assertEqual(params["temperature"], 0.0)
        self.assertEqual(params["top_k"], 0)
        self.assertEqual(params["top_p"], 0.05)

    def test_zero_temperature_selects_greedy(self) -> None:
        params = resolve_generation_params({"prompt": "p", "temperature": 0.0})
        self.assertTrue(params["greedy"])

    def test_rejects_missing_or_empty_prompt(self) -> None:
        with self.assertRaises(ValueError):
            resolve_generation_params({})
        with self.assertRaises(ValueError):
            resolve_generation_params({"prompt": "   "})
        with self.assertRaises(ValueError):
            resolve_generation_params({"prompt": 42})
        with self.assertRaises(ValueError):
            resolve_generation_params(None)

    def test_rejects_non_numeric_params(self) -> None:
        with self.assertRaises(ValueError):
            resolve_generation_params({"prompt": "p", "max_tokens": "lots"})
        with self.assertRaises(ValueError):
            resolve_generation_params({"prompt": "p", "temperature": "hot"})


class TestServerWithCheckpoint(unittest.TestCase):
    """Tests against an app backed by a tiny real checkpoint."""

    CHECKPOINT_STEP = 7

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp_dir = Path(tempfile.mkdtemp(prefix="test_server_"))
        cls.config = GPTConfig(
            vocab_size=50_257,
            context_length=16,
            n_embd=8,
            n_heads=2,
            n_layers=1,
            dropout=0.0,
        )
        cls.checkpoint_path = cls.tmp_dir / "tiny.pt"
        _write_checkpoint(cls.checkpoint_path, cls.config, step=cls.CHECKPOINT_STEP)
        cls.eval_report_path = cls.tmp_dir / "evaluation_report.json"
        cls.app = create_app(
            cls.checkpoint_path,
            device="cpu",
            eval_report_path=cls.eval_report_path,
        )
        cls.client = cls.app.test_client()

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def test_index_serves_page(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Research-Paper GPT", response.get_data(as_text=True))
        response.close()

    def test_model_info_shape(self) -> None:
        response = self.client.get("/api/model-info")
        self.assertEqual(response.status_code, 200)
        info = response.get_json()

        self.assertEqual(info["status"], "ready")
        self.assertIsNone(info["message"])
        self.assertEqual(info["device"], "cpu")
        self.assertIsNone(info["evaluation"])

        self.assertEqual(info["checkpoint"]["step"], self.CHECKPOINT_STEP)
        self.assertEqual(Path(info["checkpoint"]["path"]), self.checkpoint_path)

        self.assertEqual(info["model"]["n_layers"], self.config.n_layers)
        self.assertEqual(info["model"]["n_heads"], self.config.n_heads)
        self.assertEqual(info["model"]["n_embd"], self.config.n_embd)
        self.assertEqual(info["model"]["context_length"], self.config.context_length)
        self.assertEqual(info["model"]["vocab_size"], self.config.vocab_size)
        self.assertGreater(info["model"]["num_parameters"], 0)

        self.assertEqual(info["tokenizer"]["name"], "gpt2")
        self.assertEqual(info["tokenizer"]["backend"], "gpt2_tiktoken")
        self.assertEqual(info["tokenizer"]["vocab_size"], 50_257)
        self.assertEqual(info["tokenizer"]["eos_token_id"], 50_256)

    def test_model_info_includes_evaluation_when_report_exists(self) -> None:
        report = {
            "checkpoint_path": str(self.checkpoint_path),
            "model_size": "debug",
            "timestamp": "2026-01-01T00:00:00",
            "prompt_suite": "research_capability",
            "perplexity_results": [
                {
                    "split": "test",
                    "loss": 3.5,
                    "perplexity": 33.1,
                    "num_tokens": 1000,
                    "num_batches": 4,
                    "time_seconds": 0.5,
                }
            ],
            "skipped_splits": {},
        }
        self.eval_report_path.write_text(json.dumps(report), encoding="utf-8")
        self.addCleanup(self.eval_report_path.unlink)

        info = self.client.get("/api/model-info").get_json()
        evaluation = info["evaluation"]
        self.assertEqual(evaluation["model_size"], "debug")
        self.assertEqual(evaluation["timestamp"], "2026-01-01T00:00:00")
        self.assertEqual(evaluation["prompt_suite"], "research_capability")
        self.assertEqual(evaluation["perplexity_results"][0]["split"], "test")
        self.assertEqual(evaluation["skipped_splits"], {})

    def test_generate_streams_plain_text(self) -> None:
        torch.manual_seed(0)
        response = self.client.post(
            "/api/generate",
            json={"prompt": "Deep learning", "max_tokens": 8, "temperature": 1.0, "top_k": 0, "top_p": 1.0},
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.content_type.startswith("text/plain"))
        text = response.get_data(as_text=True)
        self.assertGreater(len(text), 0)

    def test_generate_rejects_bad_requests(self) -> None:
        response = self.client.post("/api/generate", json={})
        self.assertEqual(response.status_code, 400)
        self.assertIn("prompt", response.get_json()["error"])

        response = self.client.post("/api/generate", json={"prompt": "p", "max_tokens": "many"})
        self.assertEqual(response.status_code, 400)

        response = self.client.post("/api/generate", data="not json", content_type="text/plain")
        self.assertEqual(response.status_code, 400)

    def test_generate_busy_returns_409(self) -> None:
        state = self.app.extensions["server_state"]
        self.assertTrue(state.lock.acquire(blocking=False))
        try:
            response = self.client.post("/api/generate", json={"prompt": "hi"})
            self.assertEqual(response.status_code, 409)
        finally:
            state.lock.release()

        response = self.client.post(
            "/api/generate",
            json={"prompt": "hi", "max_tokens": 1},
        )
        self.assertEqual(response.status_code, 200)
        response.get_data()


class TestServerWithoutCheckpoint(unittest.TestCase):
    """Tests for the graceful no-checkpoint startup path."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp_dir = Path(tempfile.mkdtemp(prefix="test_server_missing_"))
        cls.app = create_app(
            cls.tmp_dir / "does_not_exist.pt",
            device="cpu",
            eval_report_path=cls.tmp_dir / "missing_report.json",
        )
        cls.client = cls.app.test_client()

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def test_model_info_reports_no_checkpoint(self) -> None:
        response = self.client.get("/api/model-info")
        self.assertEqual(response.status_code, 200)
        info = response.get_json()
        self.assertEqual(info["status"], "no_checkpoint")
        self.assertIn("run training first", info["message"])
        self.assertIsNone(info["model"])
        self.assertIsNone(info["checkpoint"])
        self.assertIsNone(info["tokenizer"])
        self.assertIsNone(info["evaluation"])

    def test_generate_returns_503(self) -> None:
        response = self.client.post("/api/generate", json={"prompt": "hi"})
        self.assertEqual(response.status_code, 503)
        self.assertIn("run training first", response.get_json()["error"])


class TestEvaluationSummaryLoader(unittest.TestCase):
    """Tests for the evaluation report summary reader."""

    def test_missing_report_returns_none(self) -> None:
        with tempfile.TemporaryDirectory(prefix="test_server_eval_") as tmp_dir:
            self.assertIsNone(load_evaluation_summary(Path(tmp_dir) / "nope.json"))

    def test_malformed_report_returns_none(self) -> None:
        with tempfile.TemporaryDirectory(prefix="test_server_eval_") as tmp_dir:
            report_path = Path(tmp_dir) / "evaluation_report.json"
            report_path.write_text("{not json", encoding="utf-8")
            self.assertIsNone(load_evaluation_summary(report_path))


if __name__ == "__main__":
    unittest.main()
