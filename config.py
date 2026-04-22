# All config in one file

VOL_TARGET = 0.15
VOL_LOOKBACK = 60          # EWM span for ex-ante daily vol
RETURN_HORIZONS = (1, 21, 63, 126, 252)
MACD_PAIRS = ((8, 24), (16, 28), (32, 96))
NUM_FEATURES = len(RETURN_HORIZONS) + len(MACD_PAIRS)  
EPS = 1e-8

# default ETF universe
DEFAULT_TICKERS = [
    # US Broad Equity
    "SPY", "QQQ", "IWM", "VTI", "MDY",

    # US Sectors
    "XLF", "XLE", "XLK", "XLI", "XLP", "XLV", "XLB", "XLU", "XLY", "KRE",

    # US Thematic
    "XOP", "XBI",

    # US Style Factors
    "VUG", "VTV",

    # Real Estate
    "VNQ",

    # International Broad
    "EFA", "EEM",

    # International Single-Country
    "EWJ", "EWG", "EWU", "EWC", "EWA", "EWZ", "EWW", "EWT", "EWS",

    # Government Bonds
    "TLT", "IEF", "SHY",

    # Credit & Aggregate
    "AGG", "LQD", "HYG", "PFF",

    # Inflation-Linked
    "TIP",

    # Commodities
    "GLD", "SLV", "DBC", "USO", "DBA",

    # Currency / FX
    "UUP", "FXE", "FXY",

    # Other / Diversifier
    "DIA", "FXI", "SMH",
]


DATA = {
    "tickers":     DEFAULT_TICKERS,
    "start":       "2005-01-01",
    "end":         "2025-12-31",
    "train_frac":  0.70,           # 70% train
    "val_frac":    0.15,           # 15% val, remaining 15% test
    "lookback":    126,            # l_t : target sequence length (≈6 months)
    "context_len": 21,             # l_c : context sequence length (≈1 month)
    "num_context": 10,             # |C| : number of context sequences per episode
    "batch_size":  64,
    "seed":        42,
}

MODEL = {
    "hidden_dim":   128,            # LSTM / attention hidden size; was 64 for Run 1
    "num_heads":    4,             # attention heads in X-Trend
    "dropout":      0.1,
    "warmup_steps": 63,            # l_s : ignore first 63 predictions in Sharpe loss
}

TRAIN = {
    "lr":            1e-3,         # Adam learning rate
    "weight_decay":  1e-4,         # L2 regularisation
    "max_grad_norm": 1.0,          # gradient clipping
    "epochs":        30,
    "patience":      10,           # early stopping patience
    "cost_bps":      5.0,          # transaction cost parameter
}

CPD = {
    "lbw":    21,       # lookback window for GP fitting
    "nu":     0.9,      # severity threshold
    "l_min":  5,        # discard regimes shorter than this
    "l_max":  21,       # cap regime length
    "recompute_every": 21,   # recompute causal CPD cache every 21 trading days
}

