"""Exact token accounting for a pinned, publicly downloadable NVIDIA tokenizer."""

from pathlib import Path
import hashlib, json, urllib.request

MODEL_ID = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
REVISION = "d468880b6ad3c6e0d21377ce7242adaea4cc884d"
TOKENIZER_SHA256 = "623c34567aebb18582765289fbe23d901c62704d6518d71866e0e58db892b5b7"


class FixtureTokenizer:
    """Character units for offline mechanics tests ONLY; never called model tokens."""

    label = "fixture_unicode_characters_NOT_model_tokens"

    def encode(self, text):
        return list(text)

    def decode(self, ids):
        return "".join(ids)


class PinnedTokenizer:
    label = MODEL_ID + "@" + REVISION

    def __init__(self, cache):
        from tokenizers import Tokenizer

        path = Path(cache) / "nemotron-tokenizer.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            url = f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/tokenizer.json"
            with urllib.request.urlopen(url, timeout=120) as response:
                blob = response.read(30_000_001)
            if len(blob) > 30_000_000:
                raise ValueError("Tokenizer exceeds expected size")
            if hashlib.sha256(blob).hexdigest() != TOKENIZER_SHA256:
                raise ValueError("Tokenizer checksum mismatch")
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(blob)
            temporary.replace(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != TOKENIZER_SHA256:
            raise ValueError("Cached tokenizer checksum mismatch")
        self.tokenizer = Tokenizer.from_file(str(path))

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False)


def get_tokenizer(mode, config):
    if mode == "fixture":
        return FixtureTokenizer()
    if config.get("tokenizer_model", MODEL_ID) != MODEL_ID:
        raise ValueError(
            "Add and validate a pinned tokenizer adapter for this model before live use"
        )
    return PinnedTokenizer(config.get("tokenizer_cache", "cache/tokenizer"))


def pad_exact(text, budget, tokenizer):
    """Preserve original text and append inert padding to an exact encode length."""
    current = len(tokenizer.encode(text))
    if current > budget:
        raise ValueError(f"Text exceeds budget: {current}>{budget}")
    if isinstance(tokenizer, FixtureTokenizer):
        return text + " " * (budget - current)
    # Repeated ` x` is one token per repetition for this pinned tokenizer after
    # a stable prefix. Re-encode the whole text and fail if exactness is impossible.
    amount = max(0, budget - current)
    candidate = text + " x" * amount
    for _ in range(32):
        actual = len(tokenizer.encode(candidate))
        if actual == budget:
            return candidate
        amount += budget - actual
        if amount < 0:
            break
        candidate = text + " x" * amount
    raise ValueError(
        "Cannot pad without changing core text; enlarge or change the slot budget"
    )
