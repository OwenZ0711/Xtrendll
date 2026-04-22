# xtrendll — end-to-end setup guide

Follow these steps once, top to bottom. After that, every subsequent experiment is just steps 4 + 5.

---

## Step 1 — put `xtrendll` into its own GitHub repo

The fork is **self-contained**: everything `xtrendll` imports lives inside this folder (the base-repo files have been copied in). You don't clone `S11685_Final_Project` anywhere.

```bash
# from anywhere outside the original checkout
cp -r /path/to/S11685_Final_Project/xtrendll ./xtrendll
cd xtrendll

git init
git add .
git commit -m "Initial commit: xtrendll self-contained fork"

# create an empty repo on GitHub named "xtrendll" (or whatever you prefer)
git remote add origin https://github.com/<YOUR-USERNAME>/xtrendll.git
git branch -M main
git push -u origin main
```

The fork's top level on GitHub should look like:

```
xtrendll/
├── README.md
├── MANUAL.md                          ← this file
├── __init__.py
├── lag_blocks.py
├── lead_lag.py
├── x_trend_ll.py
├── prep_artifacts.py
├── results_io.py
├── config.py
├── components.py
├── cpd.py
├── data.py
├── train.py
├── backtest.py
├── x_trend.py
├── x_trend_cross_section.py
├── prep_artifacts.ipynb
├── xtrendll_a1_no_bennett.ipynb
└── xtrendll_a2_with_bennett.ipynb
```

Before you push (or right after), **edit the clone cell** of the two training notebooks — `xtrendll_a1_no_bennett.ipynb` cell 2 and `xtrendll_a2_with_bennett.ipynb` cell 2 — and replace

```python
XTRENDLL_REPO = 'https://github.com/YOUR-USERNAME/xtrendll.git'
```

with your actual fork URL. Commit and push.

If your repo is **private**, make Colab clone it via a personal access token:

```python
XTRENDLL_REPO = 'https://<token>@github.com/<you>/xtrendll.git'
```

---

## Step 2 — run the data prep **locally** (one-time)

Goal: `./artifacts/` — a folder of pickles that the Colab notebooks can load directly, so CPD never reruns.

Requirements on your laptop (CPU is fine, no GPU needed):

```bash
pip install torch numpy pandas scipy yfinance tqdm matplotlib jupyter
```

Then clone your fork and run the notebook:

```bash
git clone https://github.com/<YOUR-USERNAME>/xtrendll.git
cd xtrendll
jupyter notebook prep_artifacts.ipynb
```

Inside the notebook:

1. **Cell 2 (Run config)** — edit as needed:
   - `UNIVERSE = 'tuning_21'` (the 21-ETF sweep universe, default) or `'default_50'` (full 50-ETF universe from the base project).
   - `START`, `END` — date range. Defaults to 2008-01-01 → today.
   - `BUILD_BENNETT = True` if you plan to run workflow A2 (the Bennett fit takes <1 min; safe to leave on).
   - `CPD_JOBS` — leave at `os.cpu_count()` for the fastest run.
2. **Cells 3–8** run top to bottom. Approximate timings on a modern laptop at `tuning_21` (21 ETFs, 2008-now):
   - `build_panel` (download + features): 1–2 min, network-bound.
   - Train CPD: 3–8 min.
   - Val regime cache: 10–20 min.
   - Test regime cache: 10–20 min.
   - Bennett: <1 min.
   - **Total: 25–50 min first run.** Subsequent runs hit the internal cache and finish in seconds.
   - A top-level `tqdm` bar tracks stage progress; per-ticker CPD progress prints to stdout.
3. The last cell prints the contents of `./artifacts/`. The relevant pickles are:
   ```
   panel_bundle__v1_etf21.pkl
   train_regimes__v1_etf21.pkl
   val_regime_cache__v1_etf21.pkl
   test_regime_cache__v1_etf21.pkl
   lead_lag_matrix__v1_etf21.pkl      ← A2 only
   MANIFEST.json
   _cpd_cache/                         ← hidden; you don't need to upload it
   ```

---

## Step 3 — upload artefacts to Google Drive

Target path on Drive: `MyDrive/xtrendll_artifacts_v1_etf21/` (must match `ARTIFACTS_DIR` in cell 3 of the training notebooks — edit both if you pick a different name).

Easiest method:

1. Zip locally: `zip -r xtrendll_artifacts_v1_etf21.zip artifacts -x "*/_cpd_cache/*"`.
2. Upload the zip to Drive via drive.google.com.
3. Unzip — either on Drive via a desktop client, or in a Colab scratch cell:

```python
from google.colab import drive
drive.mount('/content/drive')
import shutil
shutil.unpack_archive(
    '/content/drive/MyDrive/xtrendll_artifacts_v1_etf21.zip',
    '/content/drive/MyDrive/',
)
```

Verify on Drive that `MyDrive/xtrendll_artifacts_v1_etf21/MANIFEST.json` exists.

---

## Step 4 — run workflow A1 on Colab

1. Open `xtrendll_a1_no_bennett.ipynb` from your fork (github.com/\<you\>/xtrendll → the `.ipynb` → "Open in Colab").
2. **Runtime → Change runtime type → GPU** — any of these works:

   | GPU | Runtime tier | 30-epoch A1 time (21 ETFs, `max_peers=10`) |
   |---|---|---|
   | **T4** | Free Colab | 15–25 min |
   | **L4** | Pay-as-you-go | 8–12 min |
   | **A100 40 GB** | Pay-as-you-go / Pro+ | 4–8 min |
   | **H100 80 GB** | Enterprise tier | 3–6 min |

   The code auto-detects CUDA and will happily use any of them. A100/H100 also let you safely raise `max_peers` to 20 or even 49 (full panel) without OOM — the peer encoder becomes the bottleneck well before the lag block.

3. **Run all cells**.

Cell-by-cell:

| Cell | Purpose |
|---|---|
| 1 | GPU check — prints device + bf16 support |
| 2 | Clone `xtrendll` (only this one repo now), `pip install` deps |
| 3 | Mount Drive, verify `ARTIFACTS_DIR` exists |
| 4 | Imports — everything from `xtrendll.*` |
| 5 | Config + seed — `lag_set=(1, 5, 10, 21, 30)`, `top_k=3`, no Bennett |
| 6 | `load_artifacts(...)` from Drive — no CPD reruns on Colab |
| 7 | Build episode loaders with peers on |
| 8 | Batch shape sanity print |
| 9 | Instantiate model, print parameter count |
| 10 | Forward pass on one batch |
| 11 | Train (`fit`) — 30 epochs by default |
| 12 | Test-set evaluation + backtest + equity figure |
| 13 | `save_run(...)` → `MyDrive/results_xtrendll/xtrendll_a1__v1_etf21__<timestamp>/` |

After cell 13, all results (weights, history, predictions, daily returns, equity plot, backtest tables) are persisted on Drive and can be reloaded later via `load_run(run_dir)`.

---

## Step 5 — run workflow A2 on Colab (optional)

Identical to step 4 but open `xtrendll_a2_with_bennett.ipynb`.

Prerequisite: the artefacts folder must contain `lead_lag_matrix__<tag>.pkl` (i.e., prep was run with `BUILD_BENNETT=True`). Cell 6 asserts this and raises a clear error if missing.

Only behavioural differences from A1:
- Cell 5 sets `LL_RUN['use_bennett'] = True`.
- Cell 6 loads the Bennett payload and calls `align_S_to_asset_ids(...)` to match the model's asset_id order.
- Cell 9 passes `S_matrix=S_tensor` into `XTrendLL(...)`.
- The trained `α` (Bennett bias scalar) is reported at the end; it typically drifts from `0.1` to somewhere in `[0.05, 0.3]`.

Same runtime as A1; just one extra learnable scalar parameter.

---

## Step 6 — compare runs

In a scratch Colab cell (or any local notebook with Drive mounted):

```python
from xtrendll import compare_runs

df = compare_runs([
    '/content/drive/MyDrive/results_xtrendll/xtrendll_a1__v1_etf21__20260422_143012',
    '/content/drive/MyDrive/results_xtrendll/xtrendll_a2__v1_etf21__20260422_151137',
    # add the other agent's fixed-τ run folder here if saved in the same format
])
df
```

You'll get a DataFrame indexed by run name with: `best_epoch`, `best_val_sharpe`, `test_sharpe_gross`, `test_sharpe_net`, `test_mdd_net`, `test_avg_turnover`, `n_params`, `git_sha`, `seed`.

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `ModuleNotFoundError: No module named 'xtrendll'` on Colab | Cell 2 didn't clone or `XTRENDLL_REPO` still says `YOUR-USERNAME`. Edit that line. |
| `FileNotFoundError: Artefacts not found at /content/drive/...` | Drive path doesn't match `ARTIFACTS_DIR`. Either upload to the expected path or edit the variable. |
| `FileNotFoundError: No lead_lag_matrix in the artefacts bundle` (A2) | Prep was run with `BUILD_BENNETT=False`. Re-run the Bennett cell of `prep_artifacts.ipynb`, re-upload just `lead_lag_matrix__<tag>.pkl`, retry. |
| `assert pos.shape == ...` fails | Someone changed `DATA['lookback']` or turned peers off. Restore defaults. |
| CUDA OOM on T4 | Lower `MAX_PEERS` from 10 to 5. Lag-block memory scales with `N × L × T × H`. |
| CUDA OOM on A100 with `max_peers=49` | That's ~6 GB for the lagged `K/V` tensor alone in fp32. Either halve the batch size or use `torch.autocast('cuda', dtype=torch.bfloat16)` around the forward pass. |
| Prep notebook hangs on yfinance | Check your internet; Yahoo sometimes rate-limits. Re-run cell 3; cached results from earlier attempts are kept. |

---

## What NOT to modify

- Don't change `LagAwarePeerBlock.forward`'s output shape — `XTrendCS`'s fusion relies on it being `[B, T, H]`.
- Don't rename `_xtrendll_step` without also updating the `fit(...)` calls in both training notebooks.
- Don't change the relative imports inside copied base-repo files (`config.py`, `data.py`, etc.). They refer to sibling modules inside this package; any absolute path will break.
