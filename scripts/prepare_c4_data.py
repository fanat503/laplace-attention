"""Sterile C4 -> GPT-2-token `.bin` dataset builder.

This script downloads/streams a pinned HuggingFace dataset split, tokenizes text
with a pinned tokenizer (`tiktoken` GPT-2 by default), and writes raw int32 token
files plus JSON sidecars accepted by `FixedDataset`.

Why C4 by default:
  - public, widely used LM pretraining corpus;
  - has train/validation splits;
  - deterministic streaming order when dataset revision is pinned;
  - large enough for 14B+ token studies.

Important:
  - This script does NOT shuffle. Order is dataset streaming order.
  - Pin `--revision` for paper runs.
  - Fill `DATA_CARD.md` with dataset/tokenizer details.
  - Output `.bin` files are intentionally used instead of giant `.pt` files for
    scalable, memory-mapped training.

Example for 700M/14B run:

  python scripts/prepare_c4_data.py \
    --dataset allenai/c4 --name en --revision <PINNED_REVISION> \
    --train-tokens 14013734400 \
    --val-tokens 20000000 \
    --out-dir data

The train token count above is the *stored token ids* requirement for the current
700M/14B config, not just effective training tokens.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

# Data-prep deps are optional extras (see requirements.txt): keep --help and
# imports working on hosts without them, fail with a clear message on use.
try:
    import tiktoken
    from datasets import load_dataset
    from tqdm import tqdm
except ImportError as _e:  # pragma: no cover
    tiktoken = None
    load_dataset = None
    tqdm = None
    _IMPORT_ERROR = _e
else:
    _IMPORT_ERROR = None

# Optional fast backend (marcelroed/gigatoken): bit-identical GPT-2 tokens at
# multi-x throughput. NEVER required - tiktoken remains the default and the
# reference implementation.
try:
    import gigatoken as _gigatoken
except ImportError:  # pragma: no cover
    _gigatoken = None


def require_data_deps() -> None:
    if _IMPORT_ERROR is not None:
        raise SystemExit(
            f"Missing data-prep dependency: {_IMPORT_ERROR}\n"
            "Install with: pip install datasets tiktoken tqdm"
        )


def atomic_write_json(path: str | os.PathLike[str], payload: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.name}.tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, p)


def sample_fingerprint(path: str, *, dtype: np.dtype, num_tokens: int) -> str:
    arr = np.memmap(path, mode="r", dtype=dtype, shape=(num_tokens,))
    h = hashlib.sha256()
    h.update(str(num_tokens).encode())
    h.update(str(dtype).encode())
    window = min(8192, num_tokens)
    starts = sorted(set([0, max(0, num_tokens // 2 - window // 2), max(0, num_tokens - window)]))
    for s in starts:
        h.update(str(s).encode())
        h.update(np.asarray(arr[s : s + window]).tobytes())
    return h.hexdigest()[:24]


def file_sha256(path: str, chunk_size: int = 64 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


class BatchEncoder:
    """Backend-agnostic batch encoder with a built-in sterility cross-check.

    STERILITY DESIGN. The dataset fingerprint (sha256 over the emitted token
    bytes, recorded in the sidecar) is the ground truth: two datasets are
    interchangeable iff fingerprints match, REGARDLESS of which backend
    produced them. tiktoken stays the reference; the gigatoken path is a
    speed optimization that must be byte-equivalent - enforced two ways:
      1) verified up front on this machine (see verify_backend_equivalence);
      2) spot-checked DURING the run: one random document per batch is
         re-encoded with tiktoken and must match exactly (cost ~1/batch).
    Any mismatch aborts the run loudly - a wrong-token dataset must never
    be written silently.
    """

    def __init__(self, tokenizer_name: str, backend: str, crosscheck: bool = True):
        self.reference = tiktoken.get_encoding(tokenizer_name)
        self.backend = backend
        self.crosscheck = crosscheck and backend != "tiktoken"
        self._rng = __import__("random").Random(0)
        self.literal_fallbacks = 0
        if backend == "tiktoken":
            self._enc_batch = lambda texts: self.reference.encode_ordinary_batch(texts)
        elif backend == "gigatoken":
            if _gigatoken is None:
                raise SystemExit("--tokenizer-backend gigatoken requires: pip install gigatoken")
            if tokenizer_name != "gpt2":
                raise SystemExit("gigatoken backend is only wired for the gpt2 encoding here")
            gt_tok = _gigatoken.Tokenizer("openai-community/gpt2").as_tiktoken()
            self._enc_batch = lambda texts: gt_tok.encode_ordinary_batch(texts)
            self.backend_version = getattr(_gigatoken, "__version__", "unknown")
        else:
            raise SystemExit(f"unknown --tokenizer-backend {backend!r}")

    # GPT-2's special token literal. tiktoken's encode_ordinary treats it as
    # PLAIN TEXT (the reference semantics, shared with nanoGPT-style pipelines
    # and our own pre-gigatoken datasets); gigatoken's compat layer refuses to
    # encode it as ordinary text (a defensible design, but different).
    SPECIAL_LITERAL = "<|endoftext|>"

    def encode_batch(self, texts):
        """Hybrid policy: fast backend for literal-free documents (~100% of
        web text), REFERENCE tiktoken for the rare document containing the
        special-token literal. Semantics are therefore identical to the
        original tiktoken-only pipeline BY CONSTRUCTION - datasets produced
        before and after the fast backend existed are bit-interchangeable.
        Rejected alternatives, for the record: stripping the literal changes
        the token stream even for the tiktoken path (silent format change);
        mapping it to the real EOS id injects fake document boundaries."""
        if self.backend != "tiktoken":
            slow = [i for i, t in enumerate(texts) if self.SPECIAL_LITERAL in t]
            if slow:
                out = [None] * len(texts)
                fast_idx = [i for i in range(len(texts)) if i not in set(slow)]
                if fast_idx:
                    fast = self._enc_batch([texts[i] for i in fast_idx])
                    for i, toks in zip(fast_idx, fast, strict=True):
                        out[i] = list(toks)
                for i in slow:
                    out[i] = self.reference.encode_ordinary(texts[i])
                self.literal_fallbacks += len(slow)
                return out
        out = self._enc_batch(texts)
        if self.crosscheck and texts:
            i = self._rng.randrange(len(texts))
            ref = self.reference.encode_ordinary(texts[i])
            if list(out[i]) != list(ref):
                raise RuntimeError(
                    f"TOKENIZER MISMATCH ({self.backend} vs tiktoken) on document "
                    f"sample: {list(out[i])[:8]}... != {ref[:8]}... - aborting, "
                    "dataset would not be sterile. Re-run with --tokenizer-backend tiktoken."
                )
        return out


def verify_backend_equivalence(tokenizer_name: str, backend: str) -> None:
    """Up-front gate: the chosen backend must reproduce tiktoken bit-for-bit
    on a diverse probe set BEFORE any tokens are written."""
    if backend == "tiktoken":
        return
    enc = BatchEncoder(tokenizer_name, backend, crosscheck=False)
    probes = [
        "Hello world!",
        "Числа и юникод: 3.14159, привет, 你好, éàü, emoji 🚀🔥",
        "def f(x):\n    return x**2  # code",
        " \n\t weird   whitespace \r\n mix ",
        "a" * 5000,
        "word " * 2000,
        "<|endoftext|> literal special-token text",
    ]
    got = enc.encode_batch(probes)
    for t, g in zip(probes, got, strict=True):
        ref = enc.reference.encode_ordinary(t)
        if list(g) != list(ref):
            raise SystemExit(
                f"backend {backend!r} is NOT bit-identical to tiktoken on probe "
                f"{t[:30]!r} - refusing to build a dataset with it."
            )
    print(f"[backend] {backend} verified bit-identical to tiktoken on {len(probes)} probes")


def iter_texts(dataset_name: str, name: Optional[str], split: str, revision: Optional[str], text_field: str):
    ds = load_dataset(dataset_name, name, split=split, streaming=True, revision=revision)
    for ex in ds:
        text = ex.get(text_field)
        if isinstance(text, str) and text:
            yield text


def write_split(
    *,
    dataset_name: str,
    name: Optional[str],
    split: str,
    revision: Optional[str],
    text_field: str,
    tokenizer_name: str,
    target_tokens: int,
    out_path: str,
    add_eos: bool,
    full_hash: bool,
    backend: str = "tiktoken",
    texts=None,
) -> Dict[str, Any]:
    encoder = BatchEncoder(tokenizer_name, backend)
    enc = encoder.reference
    eos = enc.eot_token
    dtype = np.dtype("int32")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    tmp_path = f"{out_path}.tmp.{os.getpid()}"
    arr = np.memmap(tmp_path, mode="w+", dtype=dtype, shape=(target_tokens,))

    pos = 0
    docs = 0
    empty_docs = 0
    truncated_last_doc = False
    stream_hash = hashlib.sha256()
    started = time.time()

    # Batched encoding (order-preserving, hence fingerprint-preserving): the
    # token STREAM is identical to the old per-document loop - documents are
    # encoded independently in both cases and written in stream order. The
    # batch size only affects throughput, never content.
    BATCH_DOCS = 512
    pbar = tqdm(total=target_tokens, unit="tok", unit_scale=True,
                desc=f"{split} -> {Path(out_path).name}",
                mininterval=float(os.environ.get("PREP_LOG_INTERVAL_SEC", "100")),
                maxinterval=float(os.environ.get("PREP_LOG_INTERVAL_SEC", "100")),
                ascii=True)
    text_iter = iter_texts(dataset_name, name, split, revision, text_field) if texts is None else iter(texts)
    done = False
    while not done:
        batch = []
        for text in text_iter:
            batch.append(text)
            if len(batch) >= BATCH_DOCS:
                break
        if not batch:
            break
        for toks in encoder.encode_batch(batch):
            toks = list(toks)
            if add_eos:
                toks.append(eos)
            if not toks:
                empty_docs += 1
                continue
            remaining = target_tokens - pos
            if remaining <= 0:
                done = True
                break
            if len(toks) > remaining:
                toks = toks[:remaining]
                truncated_last_doc = True
            chunk = np.asarray(toks, dtype=dtype)
            arr[pos : pos + len(chunk)] = chunk
            stream_hash.update(chunk.tobytes())
            pos += len(chunk)
            docs += 1
            pbar.update(len(chunk))
            if pos >= target_tokens:
                done = True
                break
    pbar.close()
    arr.flush()
    del arr

    if pos != target_tokens:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise RuntimeError(
            f"Dataset split {split!r} ended before target_tokens: wrote={pos:,}, target={target_tokens:,}"
        )

    os.replace(tmp_path, out_path)
    meta = {
        "format": "raw_token_bin_v1",
        "dtype": "int32",
        "num_tokens": int(target_tokens),
        "dataset": dataset_name,
        "name": name,
        "split": split,
        "revision": revision,
        "text_field": text_field,
        "tokenizer": tokenizer_name,
        "vocab_size": enc.n_vocab,
        "eos_token": int(eos),
        "add_eos": bool(add_eos),
        "documents_consumed": int(docs),
        "empty_documents_skipped": int(empty_docs),
        "truncated_last_document": bool(truncated_last_doc),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_sec": float(time.time() - started),
        "tokenizer_backend": backend,
        "special_literal_fallback_docs": int(encoder.literal_fallbacks),
        "content_sha256_stream": stream_hash.hexdigest(),
        "sample_fingerprint": sample_fingerprint(out_path, dtype=dtype, num_tokens=target_tokens),
        "file_size_bytes": int(os.path.getsize(out_path)),
    }
    if full_hash:
        meta["file_sha256"] = file_sha256(out_path)
    atomic_write_json(f"{out_path}.json", meta)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="allenai/c4")
    ap.add_argument("--name", default="en")
    ap.add_argument("--revision", default=None, help="Pin this for paper runs, e.g. a HF commit hash")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--val-split", default="validation")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--tokenizer", default="gpt2")
    ap.add_argument("--train-tokens", type=int, required=True, help="Stored token ids to write for train")
    ap.add_argument("--val-tokens", type=int, default=20_000_000, help="Stored token ids to write for validation")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--train-out", default="train_fixed_tokens.bin")
    ap.add_argument("--val-out", default="val_fixed_tokens.bin")
    ap.add_argument("--no-eos", action="store_true")
    ap.add_argument("--full-hash", action="store_true", help="Compute full SHA256 of output files; slower")
    ap.add_argument("--tokenizer-backend", default="tiktoken", choices=["tiktoken", "gigatoken"],
                    help="gigatoken = multi-x faster, verified bit-identical to tiktoken "
                         "up front AND spot-checked per batch; fingerprint in the sidecar "
                         "is backend-independent ground truth")
    args = ap.parse_args()
    require_data_deps()

    if args.revision is None:
        print("WARNING: --revision not set. For NeurIPS-grade runs, pin a HuggingFace commit hash.")
    out_dir = Path(args.out_dir)
    train_path = str(out_dir / args.train_out)
    val_path = str(out_dir / args.val_out)

    verify_backend_equivalence(args.tokenizer, args.tokenizer_backend)
    common = dict(
        dataset_name=args.dataset,
        name=args.name,
        revision=args.revision,
        text_field=args.text_field,
        tokenizer_name=args.tokenizer,
        add_eos=not args.no_eos,
        full_hash=args.full_hash,
        backend=args.tokenizer_backend,
    )
    train_meta = write_split(split=args.train_split, target_tokens=args.train_tokens, out_path=train_path, **common)
    val_meta = write_split(split=args.val_split, target_tokens=args.val_tokens, out_path=val_path, **common)

    manifest = {
        "schema_version": 1,
        "dataset": args.dataset,
        "name": args.name,
        "revision": args.revision,
        "tokenizer": args.tokenizer,
        "train": train_meta,
        "val": val_meta,
    }
    atomic_write_json(str(out_dir / "prepared_dataset_manifest.json"), manifest)
    print("DONE")
    print(f"train: {train_path}")
    print(f"val  : {val_path}")
    print(f"manifest: {out_dir / 'prepared_dataset_manifest.json'}")


if __name__ == "__main__":
    main()
