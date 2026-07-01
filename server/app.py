"""Local Flask demo server for CustomLM checkpoints."""

from __future__ import annotations

import argparse
import json
import threading
from pathlib import Path
from typing import Any, Optional

import torch
from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.generate import load_model_bundle, stream_generate
from tokenization import load_runtime_tokenizer, load_tokenizer_spec_from_checkpoint

DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "final.pt"
DEFAULT_EVAL_REPORT = PROJECT_ROOT / "evaluation" / "results" / "evaluation_report.json"
MAX_NEW_TOKENS_LIMIT = 512

GENERATION_DEFAULTS = {
    "max_tokens": 200,
    "temperature": 0.8,
    "top_k": 50,
    "top_p": 0.9,
}

NO_CHECKPOINT_MESSAGE = (
    "No checkpoint yet - run training first (training/train.py), then restart the server."
)


class ServerState:
    """Model bundle loaded once at startup, shared across requests."""

    def __init__(self) -> None:
        self.model = None
        self.config = None
        self.tokenizer = None
        self.tokenizer_spec = None
        self.checkpoint_path: Optional[Path] = None
        self.checkpoint_step: Optional[int] = None
        self.device: str = "cpu"
        self.load_error: Optional[str] = None
        self.lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self.model is not None


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def load_server_state(checkpoint_path: str | Path, device: str = "auto") -> ServerState:
    state = ServerState()
    state.device = resolve_device(device)
    state.checkpoint_path = Path(checkpoint_path)

    if not state.checkpoint_path.exists():
        state.load_error = f"Checkpoint not found: {state.checkpoint_path}. {NO_CHECKPOINT_MESSAGE}"
        return state

    try:
        model, config, payload = load_model_bundle(state.checkpoint_path, device=state.device)
        model.to(state.device)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)

        tokenizer_spec = load_tokenizer_spec_from_checkpoint(state.checkpoint_path, payload=payload)
        state.tokenizer = load_runtime_tokenizer(tokenizer_spec)
        state.tokenizer_spec = tokenizer_spec
        state.model = model
        state.config = config
        step = payload.get("step")
        state.checkpoint_step = int(step) if step is not None else None
    except Exception as exc:  # noqa: BLE001 - keep the server usable without a model
        state.load_error = f"Failed to load checkpoint {state.checkpoint_path}: {exc}"
    return state


def _as_int(value: Any, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"'{name}' must be an integer.") from exc


def _as_float(value: Any, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"'{name}' must be a number.") from exc


def resolve_generation_params(payload: Any, *, vocab_size: Optional[int] = None) -> dict:
    """Validate and clamp a /api/generate request body into sampler kwargs."""
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")

    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("'prompt' must be a non-empty string.")

    max_tokens = _as_int(payload.get("max_tokens", GENERATION_DEFAULTS["max_tokens"]), "max_tokens")
    temperature = _as_float(payload.get("temperature", GENERATION_DEFAULTS["temperature"]), "temperature")
    top_k = _as_int(payload.get("top_k", GENERATION_DEFAULTS["top_k"]), "top_k")
    top_p = _as_float(payload.get("top_p", GENERATION_DEFAULTS["top_p"]), "top_p")

    temperature = min(max(temperature, 0.0), 2.0)
    top_k = max(top_k, 0)
    if vocab_size is not None:
        top_k = min(top_k, vocab_size)

    return {
        "prompt": prompt,
        "max_new_tokens": min(max(max_tokens, 1), MAX_NEW_TOKENS_LIMIT),
        "temperature": temperature,
        "top_k": top_k,
        "top_p": min(max(top_p, 0.05), 1.0),
        "greedy": temperature == 0.0,
    }


def load_evaluation_summary(report_path: str | Path = DEFAULT_EVAL_REPORT) -> Optional[dict]:
    """Read the trimmed evaluation summary shown in the UI, or None if absent."""
    report_path = Path(report_path)
    if not report_path.exists():
        return None
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(report, dict):
        return None
    return {
        "timestamp": report.get("timestamp"),
        "model_size": report.get("model_size"),
        "prompt_suite": report.get("prompt_suite"),
        "perplexity_results": report.get("perplexity_results") or [],
        "skipped_splits": report.get("skipped_splits") or {},
    }


def create_app(
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    device: str = "auto",
    *,
    eval_report_path: str | Path = DEFAULT_EVAL_REPORT,
) -> Flask:
    app = Flask(__name__)
    state = load_server_state(checkpoint_path, device=device)
    app.extensions["server_state"] = state

    @app.get("/")
    def index() -> Response:
        return send_from_directory(app.static_folder, "index.html")

    @app.get("/api/model-info")
    def model_info() -> Response:
        info: dict[str, Any] = {
            "status": "ready" if state.ready else "no_checkpoint",
            "message": state.load_error,
            "device": state.device,
            "checkpoint": None,
            "model": None,
            "tokenizer": None,
            "evaluation": load_evaluation_summary(eval_report_path),
        }
        if state.ready:
            info["checkpoint"] = {
                "path": str(state.checkpoint_path),
                "step": state.checkpoint_step,
            }
            info["model"] = {
                "n_layers": state.config.n_layers,
                "n_heads": state.config.n_heads,
                "n_embd": state.config.n_embd,
                "context_length": state.config.context_length,
                "vocab_size": state.config.vocab_size,
                "num_parameters": state.model.num_parameters(trainable_only=False),
            }
            info["tokenizer"] = {
                "name": state.tokenizer_spec.name,
                "backend": state.tokenizer_spec.backend,
                "vocab_size": state.tokenizer_spec.vocab_size,
                "eos_token_id": state.tokenizer_spec.eos_token_id,
            }
        return jsonify(info)

    @app.post("/api/generate")
    def generate_endpoint():
        if not state.ready:
            return jsonify({"error": state.load_error or NO_CHECKPOINT_MESSAGE}), 503

        try:
            params = resolve_generation_params(
                request.get_json(silent=True),
                vocab_size=state.config.vocab_size,
            )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        if not state.lock.acquire(blocking=False):
            return jsonify({"error": "A generation is already running - try again shortly."}), 409

        def stream():
            try:
                yield from stream_generate(
                    state.model,
                    state.tokenizer,
                    params["prompt"],
                    max_new_tokens=params["max_new_tokens"],
                    temperature=params["temperature"],
                    top_k=params["top_k"],
                    top_p=params["top_p"],
                    greedy=params["greedy"],
                )
            finally:
                state.lock.release()

        response = Response(
            stream_with_context(stream()),
            content_type="text/plain; charset=utf-8",
        )
        response.headers["Cache-Control"] = "no-cache"
        response.headers["X-Accel-Buffering"] = "no"
        return response

    return app


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local demo web UI for a CustomLM checkpoint")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    app = create_app(args.checkpoint, device=args.device)
    state = app.extensions["server_state"]

    if state.ready:
        print(
            f"Model loaded - {state.model.num_parameters(trainable_only=False) / 1e6:.1f}M params | "
            f"context {state.config.context_length} | tokenizer {state.tokenizer_spec.name} | "
            f"device {state.device}"
        )
    else:
        print(f"WARNING: {state.load_error}")
        print("Starting anyway - the UI will explain how to produce a checkpoint.")

    print(f"Serving demo UI at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
