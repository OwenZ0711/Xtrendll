# xtrendll

**Learnable-lag peer attention for the X-Trend cross-sectional model.**

Self-contained fork: this folder is everything you need. The code that was originally in the sibling `S11685_Final_Project` base repo (`config.py`, `components.py`, `cpd.py`, `data.py`, `train.py`, `backtest.py`, `x_trend.py`, `x_trend_cross_section.py`) has been copied into the package so it imports cleanly from a single top-level name: `xtrendll`.

## What it adds over XTrendCS

A drop-in replacement `LagAwarePeerBlock` for the undirected synchronous `CrossSectionBlock`. The new block:

1. Slices the already-encoded peer hidden states `peer_h: [B, N, T, H]` at past time offsets `τ ∈ {1, 5, 10, 21, 30}` — so target time `t` can attend to peer time `t − τ`.
2. Scores `(peer v, lag τ)` candidates with standard scaled dot-product attention.
3. Optionally adds a Bennett-style static lead-lag prior `α · log(S[v, u] + ε)` as an additive logit bias.
4. Keeps the top-k candidates per time step, softmaxes over survivors, and returns the weighted sum.

No batch-side schema changes and no edits anywhere outside this folder.

## Layout

```
xtrendll/                             # fork repo root
├── README.md
├── MANUAL.md                         # step-by-step guide (read this first)
│
├── __init__.py
├── lag_blocks.py                     # LagAwarePeerBlock
├── lead_lag.py                       # Bennett CCF-AUC S matrix + cache
├── x_trend_ll.py                     # XTrendLL(XTrendCS) + step functions
├── prep_artifacts.py                 # shared helpers + load_artifacts + CLI
├── results_io.py                     # save_run / load_run / compare_runs
│
├── config.py                         ┐
├── components.py                     │
├── cpd.py                            │  copied from the base repo so that
├── data.py                           │  the fork imports from a single
├── train.py                          │  package name (`xtrendll`).
├── backtest.py                       │
├── x_trend.py                        │
├── x_trend_cross_section.py          ┘
│
├── prep_artifacts.ipynb              # local-first data prep with tqdm
├── xtrendll_a1_no_bennett.ipynb      # Colab training — workflow A1
└── xtrendll_a2_with_bennett.ipynb    # Colab training — workflow A2
```

## Dependencies

```
torch >= 2.0, numpy, pandas, scipy, tqdm, matplotlib
yfinance   # only for prep_artifacts.ipynb (not needed on Colab training)
```

## Workflow in one picture

```
┌─────────────────────┐      ┌─────────────────────┐      ┌─────────────────────┐
│ prep_artifacts.ipynb│ ───▶ │  Google Drive       │ ───▶ │ xtrendll_a1/a2.ipynb│
│   (run locally)     │      │  artifacts/         │      │  (Colab GPU)        │
└─────────────────────┘      └─────────────────────┘      └─────────────────────┘
  1× per universe               upload once                 any GPU, any #runs
```

Any Colab GPU works — **T4, L4, A100, H100**. The two training notebooks auto-detect CUDA and will use bf16-friendly math when the GPU supports it.

See [MANUAL.md](MANUAL.md) for the click-by-click walkthrough.

## Citations

The architecture composes ideas from three papers:

* **X-Trend** — Wood et al., "X-Trend: a few-shot learning approach for cross-asset trading strategies with meta-learning." arXiv:2310.10500.
* **Cross-sectional attention** — Poh et al., "Enhancing Cross-Sectional Currency Strategies by Context-Aware Learning to Rank with Self-Attention." arXiv:2105.10019.
* **Bennett lead-lag** — Bennett, Cucuringu, Reinert, "Lead-lag detection and network clustering for multivariate time series with an application to the US equity market." arXiv:2201.08283.
* **DeltaLag** — Zhou, Wang, Cucuringu et al., "DeltaLag: Learning Dynamic Lead-Lag Patterns in Financial Markets." ICAIF '25 (arXiv:2511.00390).
