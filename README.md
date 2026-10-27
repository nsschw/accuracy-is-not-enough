# Evaluating Token Probabilities as Confidence Estimators in Human Behavior Simulations

Code for the paper [Evaluating Token Probabilities as Confidence Estimators in Human Behavior Simulations: A Case Study in Luxembourgish Next Like Prediction](https://aclanthology.org/2026.wnut-1.7/) (Schwager, Münker, Plum & Rettinger, 2026).

We test whether the token probabilities of open-weight LLMs (Ministral 3 3B and 8B) can serve as confidence estimates in a Next Like Prediction task: predicting whether a user of the Luxembourgish news platform RTL.lu likes (1) or dislikes (0) a comment, conditioned on their prior interactions. We compare off-the-shelf models with a statistical baseline, a proprietary baseline (Gemini 3.5 Flash-Lite) and task-specific supervised fine-tuning (SFT), and measure accuracy, ROC-AUC, Brier score, ECE and conditional confidence.

## Setup

```bash
uv sync
```

Create a `.env` with `WANDB_API_KEY` (for SFT) and `OPENROUTER` (for the proprietary baseline).

The RTL.lu data is **not included** due to privacy agreements. The preprocessing notebooks in `notebooks/00-preprocessing/` expect the raw comments in `data/raw/` and produce the following files in `data/processed/`:

| File | Content |
|---|---|
| `next-like-prediction-{train,test}-filtered.json` | Next Like Prediction splits (users split at user level, top 20% of users by interaction count removed) |
| `next-like-prediction-sft-train-70.json` | SFT training data in chat format |

## Usage

```bash
uv run python scripts/run_sft.py --config configs/sft/ministral-3-8b.yaml                         # SFT
uv run python scripts/run_inference.py --config configs/inference/ministral-3-8b-sft.yaml         # off-the-shelf / SFT models
uv run python scripts/run_statistical_baseline.py                                                 # statistical baseline
uv run python scripts/run_proprietary_baseline.py --config configs/inference/gemini-3.5-flash-lite.yaml  # proprietary baseline
```

Fine-tuned models are saved to `outputs/models/<model>-sft` and predictions to `outputs/results/<model>.json`. The analysis in `notebooks/01-analysis/` follows the paper:

| File | Paper |
|---|---|
| `3.2-dataset-statistics.ipynb` | §3.2, descriptive statistics |
| `4.1-model-calibration.ipynb` | §4.1 & §4.2, Table 1 & Figure 1 |
| `4.3-contextual-scaling.ipynb` | §4.3, Table 2 |

## Citation

```bibtex
@inproceedings{schwager-etal-2026-evaluating,
    title = "Evaluating Token Probabilities as Confidence Estimators in Human Behavior Simulations: A Case Study in {L}uxembourgish Next Like Prediction",
    author = {Schwager, Nils  and
      M{\"u}nker, Simon  and
      Plum, Alistair  and
      Rettinger, Achim},
    editor = "Bak, JinYeong  and
      van der Goot, Rob  and
      Jang, Hyeju  and
      Buaphet, Weerayut  and
      Ramponi, Alan  and
      Xu, Wei  and
      Ritter, Alan",
    booktitle = "Proceedings of the 11th Workshop on Natural User-generated Text",
    month = oct,
    year = "2026",
    address = "Budapest, Hungary",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.wnut-1.7/",
    pages = "89--100",
    ISBN = "979-8-89176-470-5"
}
```

## Acknowledgements

- **Data:** The comment and interaction data was provided by [RTL Lëtzebuerg](https://www.rtl.lu/).
