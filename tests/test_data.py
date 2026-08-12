# Copyright 2026 Slyatski Ilya
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.



"""Data pipeline tests: determinism, validation, sharding-compatible shapes."""
from __future__ import annotations

import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data import FixedDataset, fixed_token_collate, get_dataloader  # noqa: E402


@pytest.fixture()
def token_file(tmp_path):
    tokens = (torch.arange(0, 1000, dtype=torch.int32) % 250)
    p = str(tmp_path / "tokens.pt")
    torch.save(tokens, p)
    return p


class TestFixedDataset:
    def test_len_and_non_overlap(self, token_file):
        ds = FixedDataset(token_file, seq_len=9, expected_vocab_size=250)
        assert len(ds) == 100  # 1000 // (9+1)
        a = ds[0]["input_ids"]
        b = ds[1]["input_ids"]
        assert a.shape == (10,)
        assert int(a[-1]) + 1 == int(b[0]), "chunks must be adjacent, non-overlapping"

    def test_deterministic(self, token_file):
        ds1 = FixedDataset(token_file, seq_len=9)
        ds2 = FixedDataset(token_file, seq_len=9)
        for i in (0, 7, 42, 99):
            assert torch.equal(ds1[i]["input_ids"], ds2[i]["input_ids"])

    def test_out_of_vocab_rejected_full_scan(self, token_file):
        with pytest.raises(ValueError):
            FixedDataset(token_file, seq_len=9, expected_vocab_size=100, validate_full=True)

    def test_float_tensor_rejected(self, tmp_path):
        p = str(tmp_path / "bad.pt")
        torch.save(torch.rand(100), p)
        with pytest.raises(TypeError):
            FixedDataset(p, seq_len=9)

    def test_2d_tensor_rejected(self, tmp_path):
        p = str(tmp_path / "bad2d.pt")
        torch.save(torch.ones(10, 10, dtype=torch.int32), p)
        with pytest.raises(ValueError):
            FixedDataset(p, seq_len=3)

    def test_too_short_rejected(self, tmp_path):
        p = str(tmp_path / "short.pt")
        torch.save(torch.ones(5, dtype=torch.int32), p)
        with pytest.raises(ValueError):
            FixedDataset(p, seq_len=9)

    def test_negative_index(self, token_file):
        ds = FixedDataset(token_file, seq_len=9)
        assert torch.equal(ds[-1]["input_ids"], ds[len(ds) - 1]["input_ids"])

    def test_fingerprint_stable(self, token_file):
        f1 = FixedDataset(token_file, seq_len=9).info.sample_fingerprint
        f2 = FixedDataset(token_file, seq_len=9).info.sample_fingerprint
        assert f1 == f2


class TestRawBin:
    def test_bin_with_sidecar(self, tmp_path):
        tokens = (torch.arange(0, 500, dtype=torch.int32) % 250)
        p = str(tmp_path / "tokens.bin")
        tokens.numpy().tofile(p)
        import json
        with open(p + ".json", "w") as f:
            json.dump({"format": "raw_token_bin_v1", "dtype": "int32", "num_tokens": 500}, f)
        ds = FixedDataset(p, seq_len=9, expected_vocab_size=250)
        assert len(ds) == 50

    def test_bin_size_mismatch_rejected(self, tmp_path):
        tokens = (torch.arange(0, 500, dtype=torch.int32) % 250)
        p = str(tmp_path / "tokens.bin")
        tokens.numpy().tofile(p)
        import json
        with open(p + ".json", "w") as f:
            json.dump({"format": "raw_token_bin_v1", "dtype": "int32", "num_tokens": 400}, f)
        with pytest.raises(ValueError):
            FixedDataset(p, seq_len=9)

    def test_bin_missing_sidecar_rejected(self, tmp_path):
        p = str(tmp_path / "tokens.bin")
        torch.arange(0, 100, dtype=torch.int32).numpy().tofile(p)
        with pytest.raises(FileNotFoundError):
            FixedDataset(p, seq_len=9)


class TestDataloader:
    def test_batch_shape_dtype(self, token_file):
        dl = get_dataloader(token_file, seq_len=9, batch_size=4, drop_last=True,
                            expected_vocab_size=250)
        b = next(iter(dl))
        assert b["input_ids"].shape == (4, 10)
        assert b["input_ids"].dtype == torch.long

    def test_collate_casts_once(self):
        batch = [{"input_ids": torch.ones(10, dtype=torch.int32)} for _ in range(3)]
        out = fixed_token_collate(batch)
        assert out["input_ids"].dtype == torch.long
        assert out["input_ids"].shape == (3, 10)

    def test_order_is_sequential(self, token_file):
        dl = get_dataloader(token_file, seq_len=9, batch_size=2, drop_last=True)
        first = next(iter(dl))["input_ids"]
        assert int(first[0, 0]) == 0
        assert int(first[1, 0]) == 10


class TestTokenizerBackends:
    """Round 11 (gigatoken): the fast backend is an OPTIMIZATION, never a
    semantics change - datasets must be bit-identical across backends, and
    a lying backend must be caught before/during the write."""

    def _prep(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "prep_mod", os.path.join(ROOT, "scripts", "prepare_c4_data.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if mod.tiktoken is None:
            pytest.skip("tiktoken not installed (optional data-prep dependency)")
        return mod

    def _corpus(self, n=400):
        import random
        import string
        rng = random.Random(7)
        words = ["".join(rng.choices(string.ascii_lowercase, k=rng.randint(2, 10)))
                 for _ in range(2000)]
        return [" ".join(rng.choices(words, k=rng.randint(10, 120))) + "." for _ in range(n)]

    def test_batched_tiktoken_matches_legacy_fingerprint(self, tmp_path):
        """The batched writer must produce the same stream as the old
        per-document loop: verify against a manual reference encoding."""
        prep = self._prep()
        import tiktoken
        import numpy as np
        corpus = self._corpus()
        enc = tiktoken.get_encoding("gpt2")
        ref_stream = []
        for t in corpus:
            toks = enc.encode_ordinary(t) + [enc.eot_token]
            ref_stream.extend(toks)
            if len(ref_stream) >= 30000:
                break
        ref = np.asarray(ref_stream[:30000], dtype=np.int32)
        out = tmp_path / "t.bin"
        prep.write_split(dataset_name="local", name=None, split="train",
                         revision=None, text_field="text", tokenizer_name="gpt2",
                         target_tokens=30000, out_path=str(out), add_eos=True,
                         full_hash=False, backend="tiktoken", texts=list(corpus))
        got = np.fromfile(out, dtype=np.int32)
        assert np.array_equal(got, ref), "batched writer changed the token stream"

    def test_backends_bit_identical(self, tmp_path):
        prep = self._prep()
        if prep._gigatoken is None:
            pytest.skip("gigatoken not installed")
        import hashlib
        corpus = self._corpus()
        outs = {}
        for backend in ("tiktoken", "gigatoken"):
            out = tmp_path / f"{backend}.bin"
            meta = prep.write_split(dataset_name="local", name=None, split="train",
                                    revision=None, text_field="text",
                                    tokenizer_name="gpt2", target_tokens=20000,
                                    out_path=str(out), add_eos=True,
                                    full_hash=False, backend=backend,
                                    texts=list(corpus))
            outs[backend] = (hashlib.sha256(out.read_bytes()).hexdigest(),
                             meta["content_sha256_stream"], meta["tokenizer_backend"])
        assert outs["tiktoken"][0] == outs["gigatoken"][0], "bin files differ across backends"
        assert outs["tiktoken"][1] == outs["gigatoken"][1], "stream fingerprints differ"
        assert outs["gigatoken"][2] == "gigatoken", "sidecar must record the backend"

    def test_lying_backend_is_caught(self):
        """The per-batch cross-check must abort on any token mismatch."""
        prep = self._prep()
        enc = prep.BatchEncoder("gpt2", "tiktoken")
        enc.crosscheck = True
        enc._enc_batch = lambda texts: [[1, 2, 3] for _ in texts]  # lying backend
        import pytest as _pytest
        with _pytest.raises(RuntimeError, match="TOKENIZER MISMATCH"):
            enc.encode_batch(["hello world", "another document"])

    def test_verify_backend_equivalence_gate(self):
        prep = self._prep()
        # tiktoken backend: no-op, must not raise
        prep.verify_backend_equivalence("gpt2", "tiktoken")

    def test_special_literal_hybrid_fallback(self, tmp_path):
        """Round 12: documents containing the literal '<|endoftext|>' must be
        encoded EXACTLY like the legacy tiktoken-only pipeline (as plain
        text), on every backend. The gigatoken compat layer refuses such
        documents, so the encoder must route them to reference tiktoken -
        verified here against a manually built legacy stream."""
        prep = self._prep()
        import tiktoken
        import numpy as np
        enc = tiktoken.get_encoding("gpt2")
        corpus = ["plain doc one.",
                  "tricky <|endoftext|> in the middle",
                  "<|endoftext|>", "x<|endoftext|>y",
                  "double <|endoftext|><|endoftext|> hit",
                  "plain doc two."]
        legacy = []
        for t in corpus:
            legacy.extend(enc.encode_ordinary(t) + [enc.eot_token])
        n = len(legacy)
        ref = np.asarray(legacy, dtype=np.int32)
        backends = ["tiktoken"] + (["gigatoken"] if prep._gigatoken is not None else [])
        for backend in backends:
            out = tmp_path / f"lit_{backend}.bin"
            meta = prep.write_split(dataset_name="local", name=None, split="train",
                                    revision=None, text_field="text",
                                    tokenizer_name="gpt2", target_tokens=n,
                                    out_path=str(out), add_eos=True,
                                    full_hash=False, backend=backend,
                                    texts=list(corpus))
            got = np.fromfile(out, dtype=np.int32)
            assert np.array_equal(got, ref), f"{backend}: literal handling diverged from legacy"
            if backend == "gigatoken":
                assert meta["special_literal_fallback_docs"] == 4


class TestFixedDatasetTokenExact:
    """Round-9 audit: the old non-overlap test checked window ADJACENCY of
    the first two items only. This one replays the documented invariant
    idx*(seq_len+1) slicing for EVERY window and every tested block size,
    token-for-token, plus the tail-drop bound. A silent off-by-one in the
    stride would leak targets across windows in BOTH twins - invisible in
    loss curves, fatal for the independence claim behind the val metric."""

    def test_every_window_matches_manual_slice(self, tmp_path):
        T = torch.arange(1000, dtype=torch.long) % 250
        p = str(tmp_path / "toks.pt")
        torch.save(T, p)
        for block in (16, 64, 128):
            ds = FixedDataset(p, block, expected_vocab_size=250)
            n = len(ds)
            assert n == 1000 // (block + 1)
            for i in range(n):
                want = T[i * (block + 1):(i + 1) * (block + 1)]
                got = ds[i]["input_ids"]
                assert torch.equal(got.long(), want), f"block={block} idx={i}"
            dropped = 1000 - n * (block + 1)
            assert 0 <= dropped < block + 1
