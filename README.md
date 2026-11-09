# HiCD
HiCD: Software Vulnerability Type Classification via Hierarchical Contrastive Learning with Class Relation Distillation

## Installation

Use Python 3.10 or 3.11. Create and activate an environment first:

```bash
python -m venv .venv
```

```bash
python -m pip install -r requirements.txt
```

## Prepare data

Download the official [PrimeVul](https://github.com/DLVulDet/PrimeVul),
[MegaVul](https://github.com/Icyrockton/MegaVul), and
Then run:

```bash
python scripts/prepare_data.py --dataset primevul
python scripts/prepare_data.py --dataset megavul
```

## Train and evaluate

The experiment runner prints commands by default. Add `--execute` to train:

```bash
python -m ablations.run --config config/primevul_codebert.json --experiments full --execute
python -m ablations.run --config config/megavul_codebert.json --experiments full --execute
```

Each run creates a timestamped directory under `checkpoints/`, containing
`args.json`, `history.jsonl`, `best_model.pt`, validation/test predictions,
metrics, and `summary.json`. Select the checkpoint by validation Macro-F1.

```bash
python -m method.test_primevul --checkpoint RUN_DIR --data_dir data/processed/primevul --split test
python -m method.test_megavul --checkpoint RUN_DIR --data_dir data/processed/megavul --split test
```

## Citation and rights

```bibtex
@unpublished{wang_hicd,
  title = {HiCD: Software Vulnerability Type Classification via Hierarchical
           Contrastive Learning with Class Relation Distillation},
  author = {Wang, Jiaxi and Zhou, Yanrong and Chen, Zhikai and Wu, Longfei and Guan, Zhitao},
  note = {Research manuscript}
}
```
