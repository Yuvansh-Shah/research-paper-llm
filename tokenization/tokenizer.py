from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESEARCH_TOKENIZER_DIR = (
    PROJECT_ROOT / "data" / "tokenizers" / "research_papers_bpe32k"
)
TOKENIZER_MANIFEST_NAME = "tokenizer_manifest.json"
DEFAULT_EOS_TOKEN = "<|endoftext|>"


@dataclass
class TokenizerSpec:
    name: str
    backend: str
    vocab_size: int
    eos_token_id: int
    tokenizer_path: Optional[str] = None
    eos_token: str = DEFAULT_EOS_TOKEN
    unk_token: Optional[str] = None
    bos_token: Optional[str] = None
    pad_token: Optional[str] = None
    special_tokens: list[str] = field(default_factory=list)
    legacy: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RuntimeTokenizer:
    def __init__(self, spec: TokenizerSpec, backend: Any) -> None:
        self.spec = spec
        self.backend = backend

    @property
    def vocab_size(self) -> int:
        return self.spec.vocab_size

    @property
    def eos_token_id(self) -> int:
        return self.spec.eos_token_id

    def encode(self, text: str, *, add_eos: bool = False) -> list[int]:
        if self.spec.backend == "gpt2_tiktoken":
            token_ids = list(self.backend.encode_ordinary(text))
        elif self.spec.backend == "hf_tokenizers":
            token_ids = list(self.backend.encode(text, add_special_tokens=False).ids)
        else:  # pragma: no cover - guarded by loader
            raise ValueError(f"Unsupported tokenizer backend: {self.spec.backend}")
        if add_eos:
            token_ids.append(self.eos_token_id)
        return token_ids

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = False) -> str:
        if self.spec.backend == "gpt2_tiktoken":
            if skip_special_tokens:
                token_ids = [token_id for token_id in token_ids if token_id != self.eos_token_id]
            return self.backend.decode(token_ids)
        if self.spec.backend == "hf_tokenizers":
            return self.backend.decode(token_ids, skip_special_tokens=skip_special_tokens)
        raise ValueError(f"Unsupported tokenizer backend: {self.spec.backend}")


def default_gpt2_tokenizer_spec() -> TokenizerSpec:
    return TokenizerSpec(
        name="gpt2",
        backend="gpt2_tiktoken",
        vocab_size=50_257,
        eos_token_id=50_256,
        eos_token=DEFAULT_EOS_TOKEN,
        special_tokens=[DEFAULT_EOS_TOKEN],
        legacy=True,
    )


def _ensure_tokenizers_dependency() -> None:
    try:
        import tokenizers  # noqa: F401
    except ImportError as exc:  # pragma: no cover - dependency path
        raise RuntimeError(
            "The 'tokenizers' package is required for the domain tokenizer workflow. "
            "Install dependencies from requirements.txt."
        ) from exc


def _resolve_tokenizer_file(path: Path) -> Path:
    if path.is_dir():
        manifest_path = path / TOKENIZER_MANIFEST_NAME
        if manifest_path.exists():
            return manifest_path
        tokenizer_json = path / "tokenizer.json"
        if tokenizer_json.exists():
            return tokenizer_json
    return path


def _coerce_tokenizer_spec(payload: Any, *, base_dir: Optional[Path] = None) -> TokenizerSpec:
    if payload is None:
        return default_gpt2_tokenizer_spec()

    if isinstance(payload, TokenizerSpec):
        return payload

    if isinstance(payload, str):
        if payload == "gpt2":
            return default_gpt2_tokenizer_spec()
        candidate = Path(payload)
        if base_dir is not None and not candidate.is_absolute():
            candidate = (base_dir / candidate).resolve()
        return load_tokenizer_spec(candidate)

    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported tokenizer spec payload: {type(payload)}")

    if payload.get("backend") == "gpt2_tiktoken":
        return default_gpt2_tokenizer_spec()

    tokenizer_path = payload.get("tokenizer_path")
    if tokenizer_path and base_dir is not None:
        candidate = Path(tokenizer_path)
        if not candidate.is_absolute():
            tokenizer_path = str((base_dir / candidate).resolve())

    special_tokens = payload.get("special_tokens") or []
    return TokenizerSpec(
        name=str(payload.get("name") or "custom"),
        backend=str(payload.get("backend") or "hf_tokenizers"),
        vocab_size=int(payload["vocab_size"]),
        eos_token_id=int(payload["eos_token_id"]),
        tokenizer_path=tokenizer_path,
        eos_token=str(payload.get("eos_token") or DEFAULT_EOS_TOKEN),
        unk_token=payload.get("unk_token"),
        bos_token=payload.get("bos_token"),
        pad_token=payload.get("pad_token"),
        special_tokens=[str(token) for token in special_tokens],
        legacy=bool(payload.get("legacy", False)),
    )


def load_tokenizer_spec(source: Any, *, base_dir: Optional[Path] = None) -> TokenizerSpec:
    if source is None:
        return default_gpt2_tokenizer_spec()

    if isinstance(source, (TokenizerSpec, dict)):
        return _coerce_tokenizer_spec(source, base_dir=base_dir)

    if isinstance(source, str) and source == "gpt2":
        return default_gpt2_tokenizer_spec()

    resolved = _resolve_tokenizer_file(Path(source))
    if not resolved.exists():
        raise FileNotFoundError(f"Tokenizer artifact not found: {resolved}")

    if resolved.name == TOKENIZER_MANIFEST_NAME:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        return _coerce_tokenizer_spec(payload.get("tokenizer") or payload, base_dir=resolved.parent)

    if resolved.name == "tokenizer.json":
        _ensure_tokenizers_dependency()
        from tokenizers import Tokenizer  # noqa: PLC0415

        tokenizer = Tokenizer.from_file(str(resolved))
        vocab = tokenizer.get_vocab()
        eos_token_id = vocab.get(DEFAULT_EOS_TOKEN)
        if eos_token_id is None:
            raise ValueError(
                f"Tokenizer JSON is missing the required EOS token '{DEFAULT_EOS_TOKEN}': {resolved}"
            )
        return TokenizerSpec(
            name=resolved.parent.name,
            backend="hf_tokenizers",
            vocab_size=tokenizer.get_vocab_size(),
            eos_token_id=int(eos_token_id),
            tokenizer_path=str(resolved),
            eos_token=DEFAULT_EOS_TOKEN,
            unk_token="[UNK]" if "[UNK]" in vocab else None,
            special_tokens=[token for token in (DEFAULT_EOS_TOKEN, "[UNK]") if token in vocab],
        )

    raise ValueError(f"Unsupported tokenizer artifact: {resolved}")


def maybe_load_recommended_tokenizer_spec() -> Optional[TokenizerSpec]:
    manifest_path = DEFAULT_RESEARCH_TOKENIZER_DIR / TOKENIZER_MANIFEST_NAME
    if manifest_path.exists():
        return load_tokenizer_spec(manifest_path)
    return None


def create_runtime_tokenizer(spec: TokenizerSpec) -> RuntimeTokenizer:
    if spec.backend == "gpt2_tiktoken":
        try:
            import tiktoken  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - dependency path
            raise RuntimeError(
                "tiktoken is required for legacy GPT-2 tokenizer support. "
                "Install dependencies from requirements.txt."
            ) from exc
        return RuntimeTokenizer(spec, tiktoken.get_encoding("gpt2"))

    if spec.backend == "hf_tokenizers":
        _ensure_tokenizers_dependency()
        if not spec.tokenizer_path:
            raise ValueError("HF tokenizer spec is missing tokenizer_path.")
        from tokenizers import Tokenizer  # noqa: PLC0415

        tokenizer = Tokenizer.from_file(str(spec.tokenizer_path))
        return RuntimeTokenizer(spec, tokenizer)

    raise ValueError(f"Unsupported tokenizer backend: {spec.backend}")


def load_runtime_tokenizer(source: Any, *, base_dir: Optional[Path] = None) -> RuntimeTokenizer:
    return create_runtime_tokenizer(load_tokenizer_spec(source, base_dir=base_dir))


def load_tokenizer_spec_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    payload: Optional[dict[str, Any]] = None,
) -> TokenizerSpec:
    import torch  # noqa: PLC0415

    checkpoint_path = Path(checkpoint_path)
    if payload is None:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    tokenizer_payload = payload.get("tokenizer")
    if tokenizer_payload is None:
        return default_gpt2_tokenizer_spec()
    return _coerce_tokenizer_spec(tokenizer_payload, base_dir=checkpoint_path.parent)


def save_tokenizer_manifest(
    output_dir: str | Path,
    spec: TokenizerSpec,
    *,
    training_metadata: Optional[dict[str, Any]] = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"tokenizer": spec.to_dict()}
    if training_metadata:
        payload["training"] = training_metadata
    manifest_path = output_dir / TOKENIZER_MANIFEST_NAME
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path
