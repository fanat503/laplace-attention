"""Sterile pretraining data preparation for HLA experiments.

Downloads FineWeb-Edu (high-quality educational web text), tokenizes with
GPT-2 BPE, saves as .bin + .json sidecar compatible with src/data.py.

Properties:
    - Streaming download (no full dataset in memory).
    - Deterministic ordering (dataset is sorted by quality score).
    - GPT-2 BPE tokenization (vocab_size=50257, matches config).
    - EOT token appended to each document (standard for autoregressive LM).
    - Atomic writes.
    - SHA256 verification.
    - Reproducible (no RNG, deterministic tokenization).

Output format:
    train.bin: int32 tokens, one after another, no padding.
    train.bin.json: sidecar with metadata (matches src/data.py format).

Usage:
    python scripts/prepare_data.py --output /kaggle/working/data/train.bin --tokens 5000000000
    python scripts/prepare_data.py --output /kaggle/working/data/val.bin --tokens 50000000

    # Or use a smaller dataset for quick testing:
    python scripts/prepare_data.py --output /tmp/test.bin --tokens 1000000
"""
import os
import json
import hashlib
import argparse
from pathlib import Path

import numpy as np


# Lazy imports for Kaggle compatibility (avoid loading tiktoken/datasets at module level).
def _import_dependencies():
    import tiktoken
    from datasets import load_dataset
    return tiktoken, load_dataset


def tokenize_example(example: dict, enc, eot_token: int) -> list[int]:
    """Tokenize one example. Append EOT for document boundary."""
    text = example["text"]
    if not text:
        return []
    tokens = enc.encode_ordinary(text)
    tokens.append(eot_token)
    return tokens


def download_and_tokenize(
        output_path: str,
        target_tokens: int,
        dataset_name: str,
        dataset_split: str,
        dataset_config: str | None,
        text_field: str,
        max_examples: int | None,
        seed: int,
        show_progress_every: int,
) -> dict:
    """Stream dataset, tokenize, write to .bin + .json."""

    tiktoken, load_dataset = _import_dependencies()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_path.with_suffix(output_path.suffix + ".json")

    # Tokenizer.
    print("Loading GPT-2 tokenizer...")
    enc = tiktoken.get_encoding("gpt2")
    eot_token = enc.eot_token
    vocab_size = enc.n_vocab
    print(f"  vocab_size: {vocab_size}, eot_token: {eot_token}")

    # Load dataset (streaming).
    print(f"Loading dataset: {dataset_name} (split={dataset_split}, streaming=True)...")
    if dataset_config:
        ds = load_dataset(dataset_name, dataset_config, split=dataset_split, streaming=True)
    else:
        ds = load_dataset(dataset_name, split=dataset_split, streaming=True)

    # Write tokens to .bin (atomic).
    tmp_bin = output_path.with_suffix(output_path.suffix + ".tmp")
    token_count = 0
    example_count = 0
    sha256 = hashlib.sha256()

    print(f"Streaming and tokenizing (target: {target_tokens:,} tokens)...")
    with open(tmp_bin, "wb") as f:
        for example in ds:
            if max_examples is not None and example_count >= max_examples:
                break

            tokens = tokenize_example(example, enc, eot_token)
            if not tokens:
                continue

            # Convert to int32 array, write bytes.
            arr = np.array(tokens, dtype=np.int32)
            arr_bytes = arr.tobytes()
            f.write(arr_bytes)
            sha256.update(arr_bytes)

            token_count += len(tokens)
            example_count += 1

            # Progress.
            if example_count % show_progress_every == 0:
                pct = 100 * token_count / target_tokens
                print(
                    f"  [{example_count:,} examples, "
                    f"{token_count:,}/{target_tokens:,} tokens ({pct:.1f}%)]"
                )

            if token_count >= target_tokens:
                break

    # Atomic rename.
    os.replace(tmp_bin, output_path)

    # Verify file size.
    actual_bytes = os.path.getsize(output_path)
    expected_bytes = token_count * 4  # int32 = 4 bytes
    if actual_bytes != expected_bytes:
        raise RuntimeError(
            f"File size mismatch: actual={actual_bytes:,}, expected={expected_bytes:,}"
        )

    # Write metadata (atomic).
    metadata = {
        "format": "raw_token_bin_v1",
        "dtype": "int32",
        "num_tokens": token_count,
        "vocab_size": vocab_size,
        "eot_token": eot_token,
        "dataset": dataset_name,
        "dataset_config": dataset_config,
        "dataset_split": dataset_split,
        "text_field": text_field,
        "tokenizer": "gpt2",
        "sha256": sha256.hexdigest(),
        "target_tokens": target_tokens,
        "actual_examples": example_count,
        "seed": seed,
    }

    tmp_json = json_path.with_suffix(json_path.suffix + ".tmp")
    with open(tmp_json, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    os.replace(tmp_json, json_path)

    print("\nDone!")
    print(f"  Output: {output_path}")
    print(f"  Tokens: {token_count:,}")
    print(f"  Examples: {example_count:,}")
    print(f"  Size: {actual_bytes / (1024 ** 3):.2f} GB")
    print(f"  SHA256: {sha256.hexdigest()[:24]}...")
    print(f"  Metadata: {json_path}")

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Sterile pretraining data preparation for HLA experiments"
    )
    parser.add_argument(
        "--output", required=True, type=str,
        help="Output .bin path (e.g., /kaggle/working/data/train.bin)"
    )
    parser.add_argument(
        "--tokens", type=int, required=True,
        help="Target number of tokens to download"
    )
    parser.add_argument(
        "--dataset", default="HuggingFaceFW/fineweb-edu",
        help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--dataset-config", default=None,
        help="Dataset config (if any)"
    )
    parser.add_argument(
        "--split", default="train",
        help="Dataset split"
    )
    parser.add_argument(
        "--text-field", default="text",
        help="Field name containing text in dataset"
    )
    parser.add_argument(
        "--max-examples", type=int, default=None,
        help="Hard cap on number of examples (useful for testing)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed (for documentation, not used in streaming)"
    )
    parser.add_argument(
        "--progress-every", type=int, default=10000,
        help="Print progress every N examples"
    )

    args = parser.parse_args()

    metadata = download_and_tokenize(
        output_path=args.output,
        target_tokens=args.tokens,
        dataset_name=args.dataset,
        dataset_split=args.split,
        dataset_config=args.dataset_config,
        text_field=args.text_field,
        max_examples=args.max_examples,
        seed=args.seed,
        show_progress_every=args.progress_every,
    )

    # Print final summary.
    print("\n" + "=" * 60)
    print("Summary:")
    for k, v in metadata.items():
        print(f"  {k}: {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()
