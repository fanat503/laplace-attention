<div align="center">

# HLA: Holographic Laplace Attention

</div>

Every attention head does two jobs with one set of vectors: retrieval and transmission. They interfere. HLA gives each job its own channel.

This repository has:

1. The mechanism — one modified attention equation ([`src/model.py`](src/model.py), single file);
2. A sterile comparison: identical data order, init backbone, objective and schedule for base/HLA twins ([`docs/STERILITY.md`](docs/STERILITY.md));
3. Theory: 9 theorems incl. RoPE-commutation ([`docs/THEORY.md`](docs/THEORY.md));
4. Pre-registered experiment design H1–H5 ([`docs/EXPERIMENT_CARD.md`](docs/EXPERIMENT_CARD.md));
5. Every logged metric documented ([`docs/METRICS.md`](docs/METRICS.md)); data provenance pinned by revision + sha256 ([`docs/DATA_CARD.md`](docs/DATA_CARD.md)).



## How to start

```bash
git clone https://github.com/fanat503/Laplace-attention.git
cd Laplace-attention
pip install -r requirements.txt        # CPU is enough for tests

python -m pytest tests/ -q      
python scripts/audit_sterility.py      
```

Train base/HLA pair

```bash

python scripts/validate_configs.py --base configs/200m_base_s42.json --hla configs/200m_hla_s42.json
python src/make_init.py --shared-backbone \
    --base-config configs/200m_base_s42.json --hla-config configs/200m_hla_s42.json \
    --out-base inits/init_200m_base_s42.pt --out-hla inits/init_200m_hla_s42.pt

python scripts/prepare_c4_data.py --train-tokens 4700000000 --val-tokens 20000000 --out-dir data

python src/train_xla.py --config configs/200m_base_s42.json
python src/train_xla.py --config configs/200m_hla_s42.json

python scripts/make_ablation_configs.py \
    --base configs/200m_base_v2_s42.json --hla configs/200m_hla_v2_s42.json \
    --outdir configs/ablations_200m --seeds 42 43 44
```


## Citation

If you use this code, please cite it via [`CITATION.cff`](CITATION.cff).

**Apache-2.0** · Independent research; replications and compute support welcome — open an issue.
