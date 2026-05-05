
# ============================================================
# QFA PRIME FINANCE PLATFORM — STREAMLIT v4 INTERACTIVE
# Commodity Instrument Class + Advanced KPI Layout
#
# Features preserved and expanded:
# - Yahoo Finance only, no synthetic fallback
# - Crude Oil, Gold, Silver, Platinum, Natural Gas
# - Strict equal-length daily data preparation
# - Benchmark alignment with ^GSPC
# - Portfolio strategies:
#   Equal Weight, User Custom Weights, Inverse Volatility,
#   Max Sharpe, Min Volatility
# - PyPortfolioOpt optimization
# - Advanced KPI layout
# - Performance metrics
# - Rolling Beta
# - Tracking Error
# - VaR / CVaR
# - Drawdown
# - Correlation
# - Portfolio weights
# - QuantStats export
# - GARCH Volatility Lab
# - Info Hub + Data Quality
#
# Run:
# streamlit run app.py
# ============================================================

from __future__ import annotations

import os
import io
import sys
import warnings
import subprocess
import importlib.util
import datetime as dt
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

warnings.filterwarnings("ignore")


# ============================================================
# PACKAGE PREFLIGHT
# ============================================================

REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "streamlit": "streamlit",
    "yfinance": "yfinance",
    "plotly": "plotly",
    "arch": "arch",
    "pypfopt": "PyPortfolioOpt",
    "quantstats": "quantstats",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
}


def _pkg_available(import_name: str) -> bool:
    return importlib.util.find_spec(import_name) is not None


def ensure_required_packages(auto_install: bool = False) -> list[str]:
    missing = [pip for imp, pip in REQUIRED_PACKAGES.items() if not _pkg_available(imp)]
    if missing and auto_install:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *missing])
        except Exception as exc:
            print(f"[PACKAGE WARNING] Auto install failed: {exc}")
    return [pip for imp, pip in REQUIRED_PACKAGES.items() if not _pkg_available(imp)]


MISSING_PACKAGES = ensure_required_packages(auto_install=False)

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from arch import arch_model

try:
    import quantstats as qs
    QS_AVAILABLE = True
except Exception:
    QS_AVAILABLE = False

try:
    from pypfopt import EfficientFrontier, expected_returns, risk_models
    PYPFOPT_AVAILABLE = True
except Exception:
    PYPFOPT_AVAILABLE = False


# ============================================================
# APP CONFIG
# ============================================================

st.set_page_config(
    page_title="QFA Prime Finance Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

VERSION = "Streamlit Interactive v4.3 Institutional Transparency"
TRADING_DAYS = 252
DEFAULT_RF = 0.045
MIN_START_DATE = dt.date(2018, 1, 1)
DEFAULT_START = MIN_START_DATE
DEFAULT_END = dt.date.today()
MIN_OBSERVATIONS = 100
DEFAULT_MIN_VALID_RATIO = 0.70
DEFAULT_FFILL_LIMIT = 3

BENCHMARK_TICKER = "^GSPC"
BENCHMARK_NAME = "S&P 500 Daily Index"

COMMODITY_UNIVERSE: Dict[str, Dict[str, str]] = {
    "CL=F": {"name": "Crude Oil WTI Futures", "class": "Energy", "display": "Crude Oil"},
    "GC=F": {"name": "Gold Futures", "class": "Precious Metals", "display": "Gold"},
    "SI=F": {"name": "Silver Futures", "class": "Precious Metals", "display": "Silver"},
    "PL=F": {"name": "Platinum Futures", "class": "Precious Metals", "display": "Platinum"},
    "NG=F": {"name": "Natural Gas Futures", "class": "Energy", "display": "Natural Gas"},
}

COMMODITY_UNIVERSE_FUTURES = COMMODITY_UNIVERSE.copy()

COMMODITY_UNIVERSE_ETF = {
    "USO": {"name": "United States Oil Fund LP", "class": "Energy ETF Proxy", "display": "Crude Oil ETF Proxy"},
    "GLD": {"name": "SPDR Gold Shares", "class": "Precious Metals ETF Proxy", "display": "Gold ETF Proxy"},
    "SLV": {"name": "iShares Silver Trust", "class": "Precious Metals ETF Proxy", "display": "Silver ETF Proxy"},
    "PPLT": {"name": "abrdn Physical Platinum Shares ETF", "class": "Precious Metals ETF Proxy", "display": "Platinum ETF Proxy"},
    "UNG": {"name": "United States Natural Gas Fund LP", "class": "Energy ETF Proxy", "display": "Natural Gas ETF Proxy"},
}

UNIVERSE_MODES = {
    "Yahoo ETF Proxies — more stable for Streamlit Cloud": COMMODITY_UNIVERSE_ETF,
    "Yahoo Futures — CL=F / GC=F / SI=F / PL=F / NG=F": COMMODITY_UNIVERSE_FUTURES,
}


FUTURES_TO_PROXY_MAP = {
    "CL=F": "USO",
    "GC=F": "GLD",
    "SI=F": "SLV",
    "PL=F": "PPLT",
    "NG=F": "UNG",
}

PROXY_TRANSPARENCY_TABLE = pd.DataFrame([
    {"Exposure": "Crude Oil", "Futures Ticker": "CL=F", "ETF Proxy": "USO", "Proxy Name": "United States Oil Fund LP"},
    {"Exposure": "Gold", "Futures Ticker": "GC=F", "ETF Proxy": "GLD", "Proxy Name": "SPDR Gold Shares"},
    {"Exposure": "Silver", "Futures Ticker": "SI=F", "ETF Proxy": "SLV", "Proxy Name": "iShares Silver Trust"},
    {"Exposure": "Platinum", "Futures Ticker": "PL=F", "ETF Proxy": "PPLT", "Proxy Name": "abrdn Physical Platinum Shares ETF"},
    {"Exposure": "Natural Gas", "Futures Ticker": "NG=F", "ETF Proxy": "UNG", "Proxy Name": "United States Natural Gas Fund LP"},
])


GARCH_MODEL_OPTIONS = {
    "Fast Institutional": ["garch_t", "gjr_t", "tarch_t"],
    "Full Institutional": ["garch_normal", "garch_t", "gjr_normal", "gjr_t", "tarch_t", "egarch_t"],
    "TARCH/ZARCH Only": ["tarch_t"],
    "GARCH Student's T Only": ["garch_t"],
}


# ============================================================
# INSTITUTIONAL COLOR SYSTEM
# ============================================================
# Muted, non-rainbow palette. Used consistently across all Plotly charts.

QFA_COLORS = {
    "navy": "#0f172a",
    "slate": "#334155",
    "steel": "#475569",
    "muted_blue": "#1e3a8a",
    "muted_gold": "#b08900",
    "charcoal": "#111827",
    "gray": "#6b7280",
    "light_gray": "#e5e7eb",
    "risk_red": "#991b1b",
    "amber": "#92400e",
    "green": "#166534",
}

QFA_SEQUENCE = [
    QFA_COLORS["navy"],
    QFA_COLORS["muted_gold"],
    QFA_COLORS["steel"],
    QFA_COLORS["gray"],
    QFA_COLORS["charcoal"],
    QFA_COLORS["muted_blue"],
]

STRATEGY_COLORS = {
    "Equal Weight": QFA_COLORS["navy"],
    "Inverse Volatility": QFA_COLORS["steel"],
    "User Custom Weights": QFA_COLORS["muted_gold"],
    "Max Sharpe": QFA_COLORS["charcoal"],
    "Min Volatility": QFA_COLORS["gray"],
    BENCHMARK_NAME: "#64748b",
}

def color_for_name(name: str, idx: int = 0) -> str:
    return STRATEGY_COLORS.get(name, QFA_SEQUENCE[idx % len(QFA_SEQUENCE)])



# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>
        html, body, [class*="css"] {
            font-family: "DejaVu Sans", "Inter", "Segoe UI", sans-serif !important;
        }
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 3rem;
            max-width: 100% !important;
        }
        .qfa-hero {
            background: linear-gradient(135deg, #0f172a, #1e293b);
            color: white;
            padding: 28px 34px;
            border-radius: 24px;
            border-bottom: 5px solid #c9a227;
            box-shadow: 0 12px 32px rgba(15, 23, 42, 0.22);
            margin-bottom: 18px;
        }
        .qfa-hero h1 {
            margin: 0;
            font-size: 36px;
            font-weight: 900;
            letter-spacing: .3px;
        }
        .qfa-hero p {
            margin: 7px 0 0;
            color: #cbd5e1;
            font-size: 15px;
        }
        .qfa-note {
            background: #fff7ed;
            border-left: 5px solid #f59e0b;
            padding: 14px 16px;
            border-radius: 10px;
            color: #78350f;
            line-height: 1.55;
            margin: 12px 0 18px;
        }
        .kpi-card {
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 18px;
            padding: 17px 18px;
            min-height: 116px;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
        }
        .kpi-label {
            font-size: 11px;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: .08em;
            font-weight: 800;
        }
        .kpi-value {
            font-size: 26px;
            font-weight: 900;
            color: #0f172a;
            margin-top: 8px;
        }
        .kpi-sub {
            color: #64748b;
            font-size: 12px;
            margin-top: 6px;
        }
        .kpi-good { border-left: 6px solid #166534; }
        .kpi-warn { border-left: 6px solid #92400e; }
        .kpi-bad  { border-left: 6px solid #991b1b; }
        .kpi-neutral { border-left: 6px solid #334155; }
        .small-muted { color: #64748b; font-size: 12px; }
        .qfa-transparency-badge {
            background: #f8fafc;
            border: 1px solid #cbd5e1;
            color: #0f172a;
            padding: 10px 12px;
            border-radius: 12px;
            font-size: 13px;
            font-weight: 700;
            margin: 8px 0 14px;
        }
        .qfa-kpi-band {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 20px;
            padding: 16px;
            margin-bottom: 14px;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
        }
        .qfa-kpi-band-title {
            font-size: 13px;
            color: #334155;
            text-transform: uppercase;
            letter-spacing: .09em;
            font-weight: 900;
            margin-bottom: 10px;
        }
        .stTabs [data-baseweb="tab-list"] { gap: 8px; flex-wrap: wrap; }
        .stTabs [data-baseweb="tab"] {
            background: #ffffff;
            border-radius: 12px;
            border: 1px solid #e5e7eb;
            padding: 10px 16px;
            font-weight: 800;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# UTILS
# ============================================================

def safe_chart(fig: go.Figure):
    st.plotly_chart(fig, use_container_width=True, config={"responsive": True, "displaylogo": False})


def safe_df(df: pd.DataFrame, hide_index: bool = True):
    st.dataframe(df, use_container_width=True, hide_index=hide_index)


def fmt_pct(x: float) -> str:
    return "N/A" if pd.isna(x) else f"{x:.2%}"


def fmt_num(x: float) -> str:
    return "N/A" if pd.isna(x) else f"{x:.2f}"


def kpi_card(label: str, value: str, sub: str = "", tone: str = "neutral"):
    st.markdown(
        f"""
        <div class="kpi-card kpi-{tone}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def tone_for_return(x: float) -> str:
    if pd.isna(x): return "neutral"
    if x > 0.10: return "good"
    if x > 0.00: return "warn"
    return "bad"


def tone_for_drawdown(x: float) -> str:
    if pd.isna(x): return "neutral"
    if x > -0.15: return "good"
    if x > -0.35: return "warn"
    return "bad"


def tone_for_sharpe(x: float) -> str:
    if pd.isna(x): return "neutral"
    if x > 1.0: return "good"
    if x > 0.3: return "warn"
    return "bad"


def tone_for_te(x: float) -> str:
    if pd.isna(x): return "neutral"
    if x < 0.08: return "good"
    if x < 0.18: return "warn"
    return "bad"


def clear_cache():
    try:
        st.cache_data.clear()
        st.success("Cache cleared.")
    except Exception as exc:
        st.warning(f"Cache clear failed: {exc}")


# ============================================================
# DATA ENGINE
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def _download_yahoo_batch(tickers: Tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    return yf.download(
        tickers=list(tickers),
        start=start,
        end=end,
        auto_adjust=False,
        group_by="column",
        progress=False,
        threads=False,
        ignore_tz=True,
        repair=True,
    )


@st.cache_data(ttl=3600, show_spinner=False)
def _download_yahoo_single(ticker: str, start: str, end: str) -> pd.Series:
    raw = yf.download(
        tickers=ticker,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        threads=False,
        ignore_tz=True,
        repair=True,
    )

    if raw is None or raw.empty:
        raw = yf.Ticker(ticker).history(
            start=start,
            end=end,
            auto_adjust=False,
            actions=False,
            repair=True,
        )

    if raw is None or raw.empty:
        raise ValueError(f"{ticker}: Yahoo returned no data.")

    if "Adj Close" in raw.columns:
        s = raw["Adj Close"].copy()
    elif "Close" in raw.columns:
        s = raw["Close"].copy()
    else:
        raise ValueError(f"{ticker}: Yahoo output has no Adj Close or Close.")

    s.name = ticker
    return pd.to_numeric(s, errors="coerce")


def _extract_adjclose_from_batch(raw: pd.DataFrame, tickers: Tuple[str, ...]) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        if "Adj Close" in raw.columns.get_level_values(0):
            px = raw["Adj Close"].copy()
        elif "Close" in raw.columns.get_level_values(0):
            px = raw["Close"].copy()
        else:
            return pd.DataFrame()
    else:
        if "Adj Close" in raw.columns:
            px = raw[["Adj Close"]].copy()
            px.columns = list(tickers)[:1]
        elif "Close" in raw.columns:
            px = raw[["Close"]].copy()
            px.columns = list(tickers)[:1]
        else:
            return pd.DataFrame()
    return px


def download_yahoo_prices(tickers: Tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    """
    Robust Yahoo-only downloader:
    batch download -> individual ticker retry -> Ticker.history retry.
    No synthetic data is generated.
    """
    problems = []
    frames = []

    try:
        batch_raw = _download_yahoo_batch(tickers, start, end)
        batch_px = _extract_adjclose_from_batch(batch_raw, tickers)
    except Exception as exc:
        batch_px = pd.DataFrame()
        problems.append(f"Batch download failed: {exc}")

    if batch_px is not None and not batch_px.empty:
        batch_px.index = pd.to_datetime(batch_px.index)
        batch_px = batch_px.sort_index()
        for t in tickers:
            if t in batch_px.columns and batch_px[t].notna().sum() >= MIN_OBSERVATIONS:
                frames.append(batch_px[t].rename(t))

    done = {s.name for s in frames}

    for t in tickers:
        if t in done:
            continue
        try:
            time.sleep(0.35)
            frames.append(_download_yahoo_single(t, start, end))
        except Exception as exc:
            problems.append(f"{t}: {exc}")

    if not frames:
        raise ValueError(
            "Yahoo Finance returned no usable data for selected tickers. "
            "For commodity futures, this is commonly a Yahoo/yfinance timezone/throttling issue. "
            "Use sidebar Data Universe = 'Yahoo ETF Proxies — more stable for Streamlit Cloud', "
            "clear cache, or retry later. Details: " + " | ".join(problems[:8])
        )

    px = pd.concat(frames, axis=1)
    px = px.loc[:, ~px.columns.duplicated()]
    px.index = pd.to_datetime(px.index)
    px = px[~px.index.duplicated(keep="last")]
    px = px.sort_index()
    px = px.replace([np.inf, -np.inf], np.nan)

    for col in px.columns:
        px[col] = pd.to_numeric(px[col], errors="coerce")

    ordered = [t for t in tickers if t in px.columns]
    px = px[ordered] if ordered else px
    px = px.dropna(axis=1, how="all")

    if px.shape[1] == 0:
        raise ValueError("No usable Yahoo price series after robust retry engine.")

    return px


def prepare_price_matrix(
    prices: pd.DataFrame,
    min_valid_ratio: float,
    ffill_limit: int,
    allow_single: bool = False,
) -> pd.DataFrame:
    px = prices.copy()
    px.index = pd.to_datetime(px.index)
    px = px[~px.index.duplicated(keep="last")]
    px = px.sort_index()
    px = px.replace([np.inf, -np.inf], np.nan)

    valid_ratio = px.notna().mean()
    keep = valid_ratio[valid_ratio >= min_valid_ratio].index.tolist()
    px = px[keep]

    if px.shape[1] == 0:
        raise ValueError("No instruments passed minimum valid-data ratio.")

    px = px.ffill(limit=ffill_limit)
    px = px.dropna(axis=0, how="any")

    if px.shape[0] < MIN_OBSERVATIONS:
        raise ValueError(f"Aligned price matrix has fewer than {MIN_OBSERVATIONS} observations.")
    if (not allow_single) and px.shape[1] < 2:
        raise ValueError("At least two aligned instruments are required.")

    return px


def prepare_returns(prices: pd.DataFrame) -> pd.DataFrame:
    r = prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    if r.shape[0] < MIN_OBSERVATIONS:
        raise ValueError(f"Aligned return matrix has fewer than {MIN_OBSERVATIONS} observations.")
    return r


def align_two(a: pd.Series, b: pd.Series, name_a: str = "A", name_b: str = "B") -> Tuple[pd.Series, pd.Series]:
    df = pd.concat([a.rename(name_a), b.rename(name_b)], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) < MIN_OBSERVATIONS:
        raise ValueError(f"{name_a} and {name_b} cannot be aligned with enough observations.")
    return df[name_a], df[name_b]


def data_quality_report(raw: pd.DataFrame, aligned: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in raw.columns:
        rows.append({
            "Ticker": col,
            "Display Name": COMMODITY_UNIVERSE.get(col, {}).get("display", col),
            "Raw Observations": int(raw[col].notna().sum()),
            "Raw Missing %": float(raw[col].isna().mean()),
            "Aligned Price Obs": int(aligned[col].notna().sum()) if col in aligned.columns else 0,
            "Aligned Return Obs": int(returns[col].notna().sum()) if col in returns.columns else 0,
            "First Date": aligned.index.min().strftime("%Y-%m-%d") if col in aligned.columns and not aligned.empty else "",
            "Last Date": aligned.index.max().strftime("%Y-%m-%d") if col in aligned.columns and not aligned.empty else "",
        })
    return pd.DataFrame(rows)


# ============================================================
# METRICS
# ============================================================

def annualized_return(r: pd.Series) -> float:
    r = r.dropna()
    return (1 + r).prod() ** (TRADING_DAYS / len(r)) - 1 if len(r) else np.nan


def annualized_volatility(r: pd.Series) -> float:
    return r.dropna().std() * np.sqrt(TRADING_DAYS)


def sharpe_ratio(r: pd.Series, rf: float) -> float:
    vol = annualized_volatility(r)
    return np.nan if vol == 0 or pd.isna(vol) else (annualized_return(r) - rf) / vol


def max_drawdown(r: pd.Series) -> float:
    eq = (1 + r.dropna()).cumprod()
    dd = eq / eq.cummax() - 1
    return dd.min()


def historical_var(r: pd.Series, level: float) -> float:
    return r.dropna().quantile(1 - level)


def historical_cvar(r: pd.Series, level: float) -> float:
    x = r.dropna()
    v = x.quantile(1 - level)
    return x[x <= v].mean()


def tracking_error(p: pd.Series, b: pd.Series) -> float:
    p, b = align_two(p, b, "Portfolio", "Benchmark")
    return (p - b).std() * np.sqrt(TRADING_DAYS)


def beta_to_benchmark(p: pd.Series, b: pd.Series) -> float:
    p, b = align_two(p, b, "Portfolio", "Benchmark")
    return p.cov(b) / b.var()


def information_ratio(p: pd.Series, b: pd.Series) -> float:
    p, b = align_two(p, b, "Portfolio", "Benchmark")
    te = tracking_error(p, b)
    return (annualized_return(p) - annualized_return(b)) / te if te and not pd.isna(te) else np.nan


def rolling_tracking_error(p: pd.Series, b: pd.Series, window: int) -> pd.Series:
    p, b = align_two(p, b, "Portfolio", "Benchmark")
    return (p - b).rolling(window).std() * np.sqrt(TRADING_DAYS)


def rolling_beta(p: pd.Series, b: pd.Series, window: int) -> pd.Series:
    p, b = align_two(p, b, "Portfolio", "Benchmark")
    return p.rolling(window).cov(b) / b.rolling(window).var()


def rolling_var_cvar(r: pd.Series, window: int) -> pd.DataFrame:
    x = r.dropna()
    out = pd.DataFrame(index=x.index)
    for level in [0.95, 0.99]:
        q = 1 - level
        out[f"VaR {int(level*100)}%"] = x.rolling(window).quantile(q)
        out[f"CVaR {int(level*100)}%"] = x.rolling(window).apply(
            lambda z: z[z <= np.quantile(z, q)].mean() if len(pd.Series(z).dropna()) >= 20 else np.nan,
            raw=True,
        )
    return out



def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    lr = np.log(prices / prices.shift(1)).replace([np.inf, -np.inf], np.nan)
    return lr.dropna(axis=0, how="any")


def return_difference_series(returns: pd.DataFrame, base_ticker: str) -> pd.DataFrame:
    if base_ticker not in returns.columns:
        base_ticker = returns.columns[0]
    diff = pd.DataFrame(index=returns.index)
    for col in returns.columns:
        if col != base_ticker:
            diff[f"{col} minus {base_ticker}"] = returns[col] - returns[base_ticker]
    return diff.dropna(how="all")


def bollinger_frame(series: pd.Series, window: int = 63, n_std: float = 2.0) -> pd.DataFrame:
    s = series.dropna()
    out = pd.DataFrame(index=s.index)
    out["Value"] = s
    out["Rolling Mean"] = s.rolling(window).mean()
    out["Upper Band"] = out["Rolling Mean"] + n_std * s.rolling(window).std()
    out["Lower Band"] = out["Rolling Mean"] - n_std * s.rolling(window).std()
    out["Z-Score"] = (s - out["Rolling Mean"]) / s.rolling(window).std()
    return out


def rolling_sharpe(r: pd.Series, rf: float, window: int) -> pd.Series:
    daily_rf = rf / TRADING_DAYS
    excess = r - daily_rf
    return excess.rolling(window).mean() / excess.rolling(window).std() * np.sqrt(TRADING_DAYS)


def metrics_for_returns(return_map: Dict[str, pd.Series], benchmark: pd.Series, rf: float) -> pd.DataFrame:
    rows = []
    for name, r in return_map.items():
        p, b = align_two(r, benchmark, name, "Benchmark")
        rows.append({
            "Strategy / Instrument": name,
            "Annual Return": annualized_return(p),
            "Annual Volatility": annualized_volatility(p),
            "Sharpe": sharpe_ratio(p, rf),
            "Max Drawdown": max_drawdown(p),
            "VaR 95% Daily": historical_var(p, 0.95),
            "CVaR 95% Daily": historical_cvar(p, 0.95),
            "VaR 99% Daily": historical_var(p, 0.99),
            "CVaR 99% Daily": historical_cvar(p, 0.99),
            "Tracking Error vs Benchmark": tracking_error(p, b),
            "Beta vs Benchmark": beta_to_benchmark(p, b),
            "Information Ratio": information_ratio(p, b),
            "Skew": p.skew(),
            "Kurtosis": p.kurtosis(),
        })
    return pd.DataFrame(rows)


def interpret_strategy_table(metrics: pd.DataFrame) -> pd.DataFrame:
    def beta_text(x):
        if pd.isna(x): return "N/A"
        if x > 1.2: return "High equity sensitivity"
        if x > 0.8: return "Market-like beta"
        if x > 0.3: return "Moderate equity sensitivity"
        if x > -0.3: return "Low equity sensitivity"
        return "Negative beta"

    def te_text(x):
        if pd.isna(x): return "N/A"
        if x < 0.05: return "Low active risk"
        if x < 0.10: return "Moderate active risk"
        if x < 0.20: return "High active risk"
        return "Very high active risk"

    def dd_text(x):
        if pd.isna(x): return "N/A"
        if x > -0.10: return "Contained drawdown"
        if x > -0.25: return "Moderate drawdown"
        if x > -0.40: return "High drawdown"
        return "Severe drawdown"

    rows = []
    for _, r in metrics.iterrows():
        rows.append({
            "Strategy": r["Strategy / Instrument"],
            "Plain-English Read": (
                f"{dd_text(r['Max Drawdown'])}; {te_text(r['Tracking Error vs Benchmark'])}; "
                f"{beta_text(r['Beta vs Benchmark'])}; Sharpe {r['Sharpe']:.2f}."
            ),
            "Risk Snapshot": (
                f"VaR95 {r['VaR 95% Daily']:.2%}, CVaR95 {r['CVaR 95% Daily']:.2%}, "
                f"Vol {r['Annual Volatility']:.2%}."
            ),
        })
    return pd.DataFrame(rows)


def metric_dictionary() -> pd.DataFrame:
    return pd.DataFrame([
        {"Metric": "Annual Return", "Meaning": "Compounded yearly return.", "How to read": "Higher is better only if risk is acceptable."},
        {"Metric": "Annual Volatility", "Meaning": "Annualized fluctuation.", "How to read": "Higher means rougher return path."},
        {"Metric": "Sharpe", "Meaning": "Return per unit of risk after RF.", "How to read": "Above 1 strong; below 0 weak."},
        {"Metric": "Max Drawdown", "Meaning": "Worst peak-to-trough loss.", "How to read": "Capital pain measure."},
        {"Metric": "VaR", "Meaning": "Loss threshold at a confidence level.", "How to read": "VaR 95% is exceeded in worst 5% of days."},
        {"Metric": "CVaR", "Meaning": "Average loss beyond VaR.", "How to read": "More conservative tail-loss measure."},
        {"Metric": "Tracking Error", "Meaning": "Annualized active risk vs benchmark.", "How to read": "Low means benchmark-like; high means active/different."},
        {"Metric": "Rolling Beta", "Meaning": "Time-varying benchmark sensitivity.", "How to read": "1 is equity-like; 0 is low market sensitivity."},
        {"Metric": "Information Ratio", "Meaning": "Active return per unit TE.", "How to read": "Higher is better versus benchmark."},
    ])


# ============================================================
# STRATEGIES
# ============================================================

def equal_weight(returns: pd.DataFrame) -> Tuple[pd.Series, Dict[str, float]]:
    w = pd.Series(1 / returns.shape[1], index=returns.columns)
    return returns.dot(w).rename("Equal Weight"), w.to_dict()


def inverse_vol(returns: pd.DataFrame) -> Tuple[pd.Series, Dict[str, float]]:
    vol = returns.std().replace(0, np.nan)
    inv = 1 / vol
    w = (inv / inv.sum()).fillna(0)
    return returns.dot(w).rename("Inverse Volatility"), w.to_dict()


def custom_weight_strategy(returns: pd.DataFrame, weights: Dict[str, float]) -> Tuple[pd.Series, Dict[str, float]]:
    w = pd.Series(weights).reindex(returns.columns).fillna(0)
    if w.sum() <= 0:
        w = pd.Series(1 / returns.shape[1], index=returns.columns)
    else:
        w = w / w.sum()
    return returns.dot(w).rename("User Custom Weights"), w.to_dict()


def optimize_strategies(prices: pd.DataFrame, returns: pd.DataFrame, rf: float, max_weight: float) -> Tuple[Dict[str, pd.Series], Dict[str, Dict[str, float]], pd.DataFrame]:
    strategy_returns = {}
    weights = {}
    perf_rows = []

    if not PYPFOPT_AVAILABLE:
        return strategy_returns, weights, pd.DataFrame([{"Strategy": "PyPortfolioOpt", "Error": "Package unavailable"}])

    mu = expected_returns.mean_historical_return(prices, frequency=TRADING_DAYS)
    S = risk_models.CovarianceShrinkage(prices, frequency=TRADING_DAYS).ledoit_wolf()

    try:
        ef = EfficientFrontier(mu, S, weight_bounds=(0, max_weight))
        clean = ef.clean_weights() if False else None
        ef.max_sharpe(risk_free_rate=rf)
        clean = ef.clean_weights()
        perf = ef.portfolio_performance(risk_free_rate=rf, verbose=False)
        w = pd.Series(clean).reindex(returns.columns).fillna(0)
        strategy_returns["Max Sharpe"] = returns.dot(w).rename("Max Sharpe")
        weights["Max Sharpe"] = w.to_dict()
        perf_rows.append({"Strategy": "Max Sharpe", "Expected Return": perf[0], "Expected Volatility": perf[1], "Expected Sharpe": perf[2]})
    except Exception as exc:
        perf_rows.append({"Strategy": "Max Sharpe", "Error": str(exc)})

    try:
        ef = EfficientFrontier(mu, S, weight_bounds=(0, max_weight))
        ef.min_volatility()
        clean = ef.clean_weights()
        perf = ef.portfolio_performance(risk_free_rate=rf, verbose=False)
        w = pd.Series(clean).reindex(returns.columns).fillna(0)
        strategy_returns["Min Volatility"] = returns.dot(w).rename("Min Volatility")
        weights["Min Volatility"] = w.to_dict()
        perf_rows.append({"Strategy": "Min Volatility", "Expected Return": perf[0], "Expected Volatility": perf[1], "Expected Sharpe": perf[2]})
    except Exception as exc:
        perf_rows.append({"Strategy": "Min Volatility", "Error": str(exc)})

    return strategy_returns, weights, pd.DataFrame(perf_rows)


def build_strategy_set(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    rf: float,
    max_weight: float,
    custom_weights: Dict[str, float],
) -> Tuple[Dict[str, pd.Series], Dict[str, Dict[str, float]], pd.DataFrame, pd.DataFrame]:
    strategy_returns = {}
    strategy_weights = {}

    r, w = equal_weight(returns)
    strategy_returns["Equal Weight"] = r
    strategy_weights["Equal Weight"] = w

    r, w = inverse_vol(returns)
    strategy_returns["Inverse Volatility"] = r
    strategy_weights["Inverse Volatility"] = w

    r, w = custom_weight_strategy(returns, custom_weights)
    strategy_returns["User Custom Weights"] = r
    strategy_weights["User Custom Weights"] = w

    opt_r, opt_w, opt_perf = optimize_strategies(prices, returns, rf, max_weight)
    strategy_returns.update(opt_r)
    strategy_weights.update(opt_w)

    wdf = pd.DataFrame({"Ticker": returns.columns})
    wdf["Display Name"] = [COMMODITY_UNIVERSE.get(t, {}).get("display", t) for t in returns.columns]
    for name, w in strategy_weights.items():
        wdf[name] = pd.Series(w).reindex(returns.columns).fillna(0).values

    return strategy_returns, strategy_weights, wdf, opt_perf


# ============================================================
# GARCH
# ============================================================

@dataclass
class GarchSpec:
    key: str
    display: str
    vol: str = "GARCH"
    p: int = 1
    o: int = 0
    q: int = 1
    power: float = 2.0
    dist: str = "normal"


GARCH_SPECS = {
    "garch_normal": GarchSpec("garch_normal", "GARCH(1,1) Normal", "GARCH", 1, 0, 1, 2.0, "normal"),
    "garch_t": GarchSpec("garch_t", "GARCH(1,1) Student's T", "GARCH", 1, 0, 1, 2.0, "StudentsT"),
    "gjr_normal": GarchSpec("gjr_normal", "GJR-GARCH Normal", "GARCH", 1, 1, 1, 2.0, "normal"),
    "gjr_t": GarchSpec("gjr_t", "GJR-GARCH Student's T", "GARCH", 1, 1, 1, 2.0, "StudentsT"),
    "tarch_t": GarchSpec("tarch_t", "TARCH/ZARCH Student's T", "GARCH", 1, 1, 1, 1.0, "StudentsT"),
    "egarch_t": GarchSpec("egarch_t", "EGARCH Student's T", "EGARCH", 1, 1, 1, 2.0, "StudentsT"),
}


def _param(params: pd.Series, names: List[str]) -> float:
    for n in names:
        if n in params.index:
            return float(params[n])
    return 0.0


def fit_garch(r: pd.Series, ticker: str, spec: GarchSpec) -> Dict[str, Any]:
    x = 100 * r.dropna()
    if len(x) < 500:
        raise ValueError("Not enough observations for robust GARCH.")

    if spec.vol == "EGARCH":
        am = arch_model(x, mean="Constant", vol="EGARCH", p=spec.p, o=spec.o, q=spec.q, dist=spec.dist, rescale=False)
    else:
        am = arch_model(x, mean="Constant", vol="GARCH", p=spec.p, o=spec.o, q=spec.q, power=spec.power, dist=spec.dist, rescale=False)

    res = am.fit(update_freq=0, disp="off", show_warning=False)
    params = res.params.copy()
    alpha = _param(params, ["alpha[1]", "alpha"])
    beta = _param(params, ["beta[1]", "beta"])
    gamma = _param(params, ["gamma[1]", "gamma"])
    omega = _param(params, ["omega"])

    if spec.vol == "EGARCH":
        persistence = beta
    else:
        persistence = alpha + beta + 0.5 * gamma if spec.o > 0 else alpha + beta

    half_life = np.log(0.5) / np.log(persistence) if 0 < persistence < 1 else np.nan

    try:
        f = res.forecast(horizon=21, reindex=False)
        var_path = f.variance.iloc[-1]
        vol1 = float(np.sqrt(var_path.iloc[0]))
        vol5 = float(np.sqrt(var_path.iloc[:5].mean()))
        vol21 = float(np.sqrt(var_path.iloc[:21].mean()))
    except Exception:
        vol1 = vol5 = vol21 = np.nan

    return {
        "Ticker": ticker,
        "Instrument": COMMODITY_UNIVERSE.get(ticker, {}).get("display", ticker),
        "Model Key": spec.key,
        "Model": spec.display,
        "AIC": float(res.aic),
        "BIC": float(res.bic),
        "Log Likelihood": float(res.loglikelihood),
        "Omega": omega,
        "Alpha": alpha,
        "Gamma": gamma,
        "Beta": beta,
        "Persistence": persistence,
        "Half-Life Days": half_life,
        "1D Forecast Vol %": vol1,
        "5D Forecast Vol %": vol5,
        "21D Forecast Vol %": vol21,
        "Conditional Vol": res.conditional_volatility.copy(),
        "Std Resid": res.std_resid.copy(),
        "Returns %": x,
        "Summary": str(res.summary()),
    }


@st.cache_data(ttl=3600, show_spinner=False)
def run_garch_suite(returns: pd.DataFrame, model_keys: Tuple[str, ...]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, Any]]]:
    rows = []
    best_rows = []
    store = {}

    for ticker in returns.columns:
        packs = []
        for key in model_keys:
            try:
                p = fit_garch(returns[ticker], ticker, GARCH_SPECS[key])
                packs.append(p)
                store[f"{ticker}__{key}"] = p
            except Exception as exc:
                rows.append({
                    "Ticker": ticker,
                    "Instrument": COMMODITY_UNIVERSE.get(ticker, {}).get("display", ticker),
                    "Model": GARCH_SPECS[key].display,
                    "Status": f"Failed: {exc}",
                })

        packs = sorted(packs, key=lambda z: (z["BIC"], z["AIC"]))
        for rank, p in enumerate(packs, 1):
            row = {k: v for k, v in p.items() if k not in ["Conditional Vol", "Std Resid", "Returns %", "Summary"]}
            row["Rank by BIC"] = rank
            row["Status"] = "OK"
            rows.append(row)

        if packs:
            b = packs[0]
            best_rows.append({
                "Ticker": ticker,
                "Instrument": b["Instrument"],
                "Best Model": b["Model"],
                "Best Model Key": b["Model Key"],
                "BIC": b["BIC"],
                "Persistence": b["Persistence"],
                "Half-Life Days": b["Half-Life Days"],
                "1D Forecast Vol %": b["1D Forecast Vol %"],
                "5D Forecast Vol %": b["5D Forecast Vol %"],
                "21D Forecast Vol %": b["21D Forecast Vol %"],
            })

    return pd.DataFrame(rows), pd.DataFrame(best_rows), store


# ============================================================
# CHARTS
# ============================================================

def layout(fig: go.Figure, title: str, height: int = 650) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(color=QFA_COLORS["charcoal"], size=20)),
        template="plotly_white",
        height=height,
        autosize=True,
        margin=dict(l=35, r=35, t=75, b=45),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        font=dict(family="DejaVu Sans, Arial, sans-serif", size=13, color=QFA_COLORS["charcoal"]),
        colorway=QFA_SEQUENCE,
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    fig.update_xaxes(
        gridcolor="#e5e7eb",
        zerolinecolor="#cbd5e1",
        linecolor="#cbd5e1",
        tickfont=dict(color=QFA_COLORS["slate"]),
        titlefont=dict(color=QFA_COLORS["slate"]),
    )
    fig.update_yaxes(
        gridcolor="#e5e7eb",
        zerolinecolor="#cbd5e1",
        linecolor="#cbd5e1",
        tickfont=dict(color=QFA_COLORS["slate"]),
        titlefont=dict(color=QFA_COLORS["slate"]),
    )
    return fig


def fig_prices(prices: pd.DataFrame) -> go.Figure:
    norm = prices / prices.iloc[0] * 100
    fig = go.Figure()
    for col in norm.columns:
        fig.add_trace(go.Scatter(x=norm.index, y=norm[col], mode="lines", name=COMMODITY_UNIVERSE.get(col, {}).get("display", col), line=dict(color=QFA_SEQUENCE[list(norm.columns).index(col) % len(QFA_SEQUENCE)], width=2.4)))
    fig.update_yaxes(title="Base 100")
    fig.update_xaxes(title="Date")
    return layout(fig, "Normalized Commodity Prices — Base 100")


def fig_strategy_cumulative(strategy_returns: Dict[str, pd.Series], benchmark: pd.Series, selected: List[str]) -> go.Figure:
    fig = go.Figure()
    for name in selected:
        if name in strategy_returns:
            r = strategy_returns[name]
            eq = (1 + r.dropna()).cumprod()
            fig.add_trace(go.Scatter(x=eq.index, y=eq.values, mode="lines", name=name, line=dict(width=2.8, color=color_for_name(name, selected.index(name) if name in selected else 0))))
    b = (1 + benchmark.dropna()).cumprod()
    fig.add_trace(go.Scatter(x=b.index, y=b.values, mode="lines", name=BENCHMARK_NAME, line=dict(width=2.4, dash="dash", color=STRATEGY_COLORS[BENCHMARK_NAME])))
    fig.update_yaxes(title="Growth of $1")
    return layout(fig, "Strategy Cumulative Return vs Benchmark", 700)


def fig_drawdowns(strategy_returns: Dict[str, pd.Series], benchmark: pd.Series, selected: List[str]) -> go.Figure:
    def dd(r):
        eq = (1 + r.dropna()).cumprod()
        return eq / eq.cummax() - 1
    fig = go.Figure()
    for name in selected:
        if name in strategy_returns:
            d = dd(strategy_returns[name])
            fig.add_trace(go.Scatter(x=d.index, y=d.values, mode="lines", name=name, line=dict(color=color_for_name(name, selected.index(name) if name in selected else 0), width=2.2)))
    bd = dd(benchmark)
    fig.add_trace(go.Scatter(x=bd.index, y=bd.values, mode="lines", name=BENCHMARK_NAME, line=dict(dash="dash", color=STRATEGY_COLORS[BENCHMARK_NAME], width=2.2)))
    fig.update_yaxes(title="Drawdown", tickformat=".0%")
    return layout(fig, "Strategy Drawdown Comparison", 700)


def fig_weights(weights_df: pd.DataFrame, selected: List[str]) -> go.Figure:
    fig = go.Figure()
    for col in selected:
        if col in weights_df.columns:
            fig.add_trace(go.Bar(x=weights_df["Display Name"], y=weights_df[col], name=col, marker_color=color_for_name(col, selected.index(col) if col in selected else 0)))
    fig.update_yaxes(title="Weight", tickformat=".0%")
    return layout(fig, "Portfolio Weights by Strategy", 680)


def fig_risk_return(metrics: pd.DataFrame) -> go.Figure:
    df = metrics.copy()
    fig = px.scatter(
        df,
        x="Annual Volatility",
        y="Annual Return",
        color="Strategy / Instrument",
        size=(df["Sharpe"].clip(lower=0).fillna(0.01) + 0.05),
        hover_data=["Sharpe", "Max Drawdown", "Tracking Error vs Benchmark", "Beta vs Benchmark"],
        template="plotly_white",
        color_discrete_sequence=QFA_SEQUENCE,
    )
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(tickformat=".0%")
    return layout(fig, "Risk / Return Map", 680)


def fig_performance_bars(metrics: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=2, subplot_titles=["Annual Return", "Annual Volatility", "Sharpe", "Max Drawdown"])
    df = metrics.copy()
    x = df["Strategy / Instrument"]
    fig.add_trace(go.Bar(x=x, y=df["Annual Return"], name="Annual Return", marker_color=QFA_COLORS["navy"]), row=1, col=1)
    fig.add_trace(go.Bar(x=x, y=df["Annual Volatility"], name="Annual Volatility", marker_color=QFA_COLORS["steel"]), row=1, col=2)
    fig.add_trace(go.Bar(x=x, y=df["Sharpe"], name="Sharpe", marker_color=QFA_COLORS["muted_gold"]), row=2, col=1)
    fig.add_trace(go.Bar(x=x, y=df["Max Drawdown"], name="Max Drawdown", marker_color=QFA_COLORS["risk_red"]), row=2, col=2)
    fig.update_yaxes(tickformat=".0%", row=1, col=1)
    fig.update_yaxes(tickformat=".0%", row=1, col=2)
    fig.update_yaxes(tickformat=".0%", row=2, col=2)
    return layout(fig, "Performance Metrics Dashboard", 820)


def fig_te_beta(strategy_returns: Dict[str, pd.Series], benchmark: pd.Series, selected: List[str], window: int, te_target: float, te_band: float) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=["Rolling Tracking Error", "Rolling Beta"])
    for name in selected:
        if name in strategy_returns:
            te = rolling_tracking_error(strategy_returns[name], benchmark, window)
            beta = rolling_beta(strategy_returns[name], benchmark, window)
            fig.add_trace(go.Scatter(x=te.index, y=te.values, mode="lines", name=f"{name} TE", line=dict(color=color_for_name(name, selected.index(name) if name in selected else 0), width=2.2)), row=1, col=1)
            fig.add_trace(go.Scatter(x=beta.index, y=beta.values, mode="lines", name=f"{name} Beta", line=dict(color=color_for_name(name, selected.index(name) if name in selected else 0), width=2.2)), row=2, col=1)
    fig.add_hline(y=te_target, line_dash="dash", annotation_text="TE Target", row=1, col=1)
    fig.add_hline(y=te_target + te_band, line_dash="dot", annotation_text="Upper Band", row=1, col=1)
    fig.add_hline(y=max(te_target - te_band, 0), line_dash="dot", annotation_text="Lower Band", row=1, col=1)
    fig.add_hline(y=1.0, line_dash="dash", annotation_text="Beta = 1", row=2, col=1)
    fig.update_yaxes(tickformat=".0%", row=1, col=1)
    return layout(fig, "Rolling Tracking Error and Rolling Beta", 820)


def fig_te_beta_map(metrics: pd.DataFrame) -> go.Figure:
    df = metrics.copy()
    fig = px.scatter(
        df,
        x="Tracking Error vs Benchmark",
        y="Beta vs Benchmark",
        color="Strategy / Instrument",
        size=(df["Annual Volatility"].abs().fillna(0.01) + 0.01),
        hover_data=["Annual Return", "Sharpe", "Max Drawdown", "Information Ratio"],
        template="plotly_white",
        color_discrete_sequence=QFA_SEQUENCE,
    )
    fig.update_xaxes(tickformat=".0%")
    return layout(fig, "Benchmark-Relative Map — Tracking Error vs Beta", 680)


def fig_var_cvar(strategy_returns: Dict[str, pd.Series], selected: List[str], window: int) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=["Rolling VaR 95%", "Rolling CVaR 95%"])
    for name in selected:
        if name in strategy_returns:
            risk = rolling_var_cvar(strategy_returns[name], window)
            fig.add_trace(go.Scatter(x=risk.index, y=risk["VaR 95%"], mode="lines", name=f"{name} VaR", line=dict(color=color_for_name(name, selected.index(name) if name in selected else 0), width=2.2)), row=1, col=1)
            fig.add_trace(go.Scatter(x=risk.index, y=risk["CVaR 95%"], mode="lines", name=f"{name} CVaR", line=dict(color=color_for_name(name, selected.index(name) if name in selected else 0), width=2.2)), row=2, col=1)
    fig.update_yaxes(tickformat=".1%", row=1, col=1)
    fig.update_yaxes(tickformat=".1%", row=2, col=1)
    return layout(fig, "Rolling VaR / CVaR by Strategy", 800)


def fig_tail_bars(metrics: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for col in ["VaR 95% Daily", "CVaR 95% Daily", "VaR 99% Daily", "CVaR 99% Daily"]:
        fig.add_trace(go.Bar(x=metrics["Strategy / Instrument"], y=metrics[col], name=col, marker_color=QFA_SEQUENCE[["VaR 95% Daily", "CVaR 95% Daily", "VaR 99% Daily", "CVaR 99% Daily"].index(col) % len(QFA_SEQUENCE)]))
    fig.update_yaxes(tickformat=".1%")
    return layout(fig, "Tail Risk Comparison — VaR / CVaR", 680)


def fig_corr(returns: pd.DataFrame) -> go.Figure:
    corr = returns.corr()
    labels = [COMMODITY_UNIVERSE.get(c, {}).get("display", c) for c in corr.columns]
    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=labels, y=labels, colorscale=[[0, "#0f172a"], [0.5, "#f8fafc"], [1, "#b08900"]], zmid=0,
        text=np.round(corr.values, 2), texttemplate="%{text}", colorbar=dict(title="Correlation")
    ))
    return layout(fig, "Commodity Return Correlation Matrix", 650)



def fig_log_return_difference(log_ret: pd.DataFrame, base_ticker: str) -> go.Figure:
    diff = return_difference_series(log_ret, base_ticker)
    fig = go.Figure()
    for i, col in enumerate(diff.columns):
        fig.add_trace(go.Scatter(
            x=diff.index,
            y=diff[col],
            mode="lines",
            name=col,
            line=dict(color=QFA_SEQUENCE[i % len(QFA_SEQUENCE)], width=1.8),
        ))
    fig.update_yaxes(title="Daily Log Return Difference")
    return layout(fig, f"ETF / Instrument Log Return Difference vs {base_ticker}", 720)


def fig_log_return_difference_bollinger(log_ret: pd.DataFrame, spread_name: str, window: int, n_std: float) -> go.Figure:
    if spread_name not in log_ret.columns:
        base = log_ret.columns[0]
        diff = return_difference_series(log_ret, base)
        if diff.empty:
            return layout(go.Figure(), "Log Return Difference Bollinger Bands")
        s = diff.iloc[:, 0]
    else:
        s = log_ret[spread_name]

    bb = bollinger_frame(s, window=window, n_std=n_std)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bb.index, y=bb["Value"], mode="lines", name="Difference", line=dict(color=QFA_COLORS["navy"], width=1.8)))
    fig.add_trace(go.Scatter(x=bb.index, y=bb["Rolling Mean"], mode="lines", name="Rolling Mean", line=dict(color=QFA_COLORS["muted_gold"], width=2.2)))
    fig.add_trace(go.Scatter(x=bb.index, y=bb["Upper Band"], mode="lines", name="Upper Band", line=dict(color=QFA_COLORS["slate"], width=1.2, dash="dash")))
    fig.add_trace(go.Scatter(x=bb.index, y=bb["Lower Band"], mode="lines", name="Lower Band", line=dict(color=QFA_COLORS["slate"], width=1.2, dash="dash"), fill="tonexty", fillcolor="rgba(15, 23, 42, 0.06)"))
    fig.update_yaxes(title="Log Return Difference")
    return layout(fig, f"Bollinger Bands — {spread_name}", 760)


def fig_rolling_sharpe(strategy_returns: Dict[str, pd.Series], selected: List[str], rf: float, window: int) -> go.Figure:
    fig = go.Figure()
    for i, name in enumerate(selected):
        if name in strategy_returns:
            rs = rolling_sharpe(strategy_returns[name], rf, window)
            fig.add_trace(go.Scatter(x=rs.index, y=rs.values, mode="lines", name=name, line=dict(color=color_for_name(name, i), width=2.0)))
    fig.add_hline(y=0, line_dash="dash", annotation_text="Sharpe = 0")
    fig.update_yaxes(title="Rolling Sharpe")
    return layout(fig, f"{window}-Day Rolling Sharpe by Strategy", 720)


def fig_underwater_recovery(strategy_returns: Dict[str, pd.Series], selected: List[str]) -> go.Figure:
    fig = go.Figure()
    for i, name in enumerate(selected):
        if name not in strategy_returns:
            continue
        r = strategy_returns[name].dropna()
        eq = (1 + r).cumprod()
        dd = eq / eq.cummax() - 1
        fig.add_trace(go.Scatter(x=dd.index, y=dd.values, mode="lines", name=name, line=dict(color=color_for_name(name, i), width=2.0)))
    fig.update_yaxes(title="Underwater Drawdown", tickformat=".0%")
    return layout(fig, "Underwater Chart — Strategy Recovery Profile", 720)


def fig_garch_best(best: pd.DataFrame, store: Dict[str, Dict[str, Any]]) -> go.Figure:
    fig = go.Figure()
    for _, row in best.iterrows():
        key = f"{row['Ticker']}__{row['Best Model Key']}"
        if key in store:
            v = store[key]["Conditional Vol"]
            fig.add_trace(go.Scatter(x=v.index, y=v.values, mode="lines", name=f"{row['Instrument']} — {row['Best Model']}"))
    fig.update_yaxes(title="Daily Conditional Volatility (%)")
    return layout(fig, "Best-BIC GARCH Conditional Volatility", 720)


def fig_garch_rank(garch: pd.DataFrame) -> go.Figure:
    df = garch[garch["Status"] == "OK"].copy()
    if df.empty:
        return layout(go.Figure(), "GARCH Ranking Unavailable")
    fig = px.bar(df, x="Instrument", y="BIC", color="Model", barmode="group", template="plotly_white")
    return layout(fig, "GARCH Model Ranking by BIC — Lower is Better", 680)


def fig_garch_resid(ticker: str, model_key: str, store: Dict[str, Dict[str, Any]]) -> go.Figure:
    key = f"{ticker}__{model_key}"
    fig = make_subplots(rows=2, cols=2, subplot_titles=["Returns %", "Conditional Vol", "Std Residuals", "Squared Std Residuals"])
    if key not in store:
        return layout(fig, "GARCH Diagnostics Unavailable")
    p = store[key]
    r = p["Returns %"]
    v = p["Conditional Vol"]
    resid = p["Std Resid"].dropna()
    fig.add_trace(go.Scatter(x=r.index, y=r.values, mode="lines", name="Returns %"), row=1, col=1)
    fig.add_trace(go.Scatter(x=v.index, y=v.values, mode="lines", name="Cond Vol"), row=1, col=2)
    fig.add_trace(go.Scatter(x=resid.index, y=resid.values, mode="lines", name="Std Resid"), row=2, col=1)
    fig.add_trace(go.Scatter(x=resid.index, y=(resid**2).values, mode="lines", name="Sq Resid"), row=2, col=2)
    return layout(fig, f"{COMMODITY_UNIVERSE.get(ticker, {}).get('display', ticker)} — GARCH Diagnostics", 850)


# ============================================================
# EXPORTS
# ============================================================

def df_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def returns_csv(series_map: Dict[str, pd.Series]) -> bytes:
    return pd.concat(series_map, axis=1).to_csv().encode("utf-8")


def quantstats_html(strategy_name: str, r: pd.Series, rf: float) -> Optional[bytes]:
    if not QS_AVAILABLE:
        return None
    path = f"qfa_quantstats_{strategy_name.lower().replace(' ', '_')}.html"
    try:
        qs.reports.html(r.dropna(), benchmark=None, rf=rf, output=path, title=f"QFA Prime QuantStats — {strategy_name}")
        with open(path, "rb") as f:
            data = f.read()
        try:
            os.remove(path)
        except Exception:
            pass
        return data
    except Exception:
        return None


# ============================================================
# UI
# ============================================================

def main():
    global COMMODITY_UNIVERSE
    st.markdown(
        f"""
        <div class="qfa-hero">
            <h1>QFA Prime Finance Platform</h1>
            <p>Commodity Instrument Class — Institutional Interactive Strategy Lab</p>
            <p>{VERSION} • Institutional muted theme • robust Yahoo downloader • ETF proxy/futures modes • Optimization • Risk • GARCH • QuantStats</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if MISSING_PACKAGES:
        st.error("Missing packages: " + ", ".join(MISSING_PACKAGES))
        st.stop()

    with st.sidebar:
        st.header("Portfolio Gate")
        st.caption("Interactive controls")

        if st.button("Clear cache"):
            clear_cache()

        universe_mode = st.selectbox(
            "Data Universe",
            options=list(UNIVERSE_MODES.keys()),
            index=0,
            help="ETF proxy mode uses real Yahoo-listed ETFs and is more stable on Streamlit Cloud. Futures mode uses CL=F, GC=F, SI=F, PL=F, NG=F.",
        )
        active_universe = UNIVERSE_MODES[universe_mode]
        COMMODITY_UNIVERSE = active_universe

        if "ETF Proxies" in universe_mode:
            st.markdown(
                "<div class='qfa-transparency-badge'>ETF PROXY MODE ACTIVE — real Yahoo ETFs are used, not futures and not synthetic data.</div>",
                unsafe_allow_html=True,
            )
            with st.expander("Show proxy mapping", expanded=True):
                st.dataframe(PROXY_TRANSPARENCY_TABLE, use_container_width=True, hide_index=True)
        else:
            st.markdown(
                "<div class='qfa-transparency-badge'>FUTURES MODE ACTIVE — Yahoo futures tickers are used directly.</div>",
                unsafe_allow_html=True,
            )

        selected_display = st.multiselect(
            "Commodity instruments",
            options=[v["display"] for v in active_universe.values()],
            default=[v["display"] for v in active_universe.values()],
        )
        display_to_ticker = {v["display"]: k for k, v in active_universe.items()}
        selected_tickers = [display_to_ticker[d] for d in selected_display]
        st.caption("No synthetic fallback: ETF proxy mode still uses real Yahoo Finance instruments.")

        start_date = st.date_input("Start date", DEFAULT_START, min_value=MIN_START_DATE, max_value=DEFAULT_END)
        end_date = st.date_input("End date", DEFAULT_END, min_value=MIN_START_DATE, max_value=DEFAULT_END)

        rf = st.number_input("Risk-free rate", 0.0, 0.30, DEFAULT_RF, 0.005, format="%.3f")
        max_weight = st.slider("Optimization max single weight", 0.20, 1.00, 0.45, 0.05)

        st.divider()
        st.subheader("Data Discipline")
        min_valid_ratio = st.slider("Minimum valid data ratio", 0.50, 0.95, DEFAULT_MIN_VALID_RATIO, 0.05)
        ffill_limit = st.slider("Forward-fill limit", 0, 7, DEFAULT_FFILL_LIMIT, 1)

        st.divider()
        st.subheader("Custom Portfolio Weights")
        custom_raw = {}
        for ticker in selected_tickers:
            label = COMMODITY_UNIVERSE[ticker]["display"]
            custom_raw[ticker] = st.number_input(f"{label} weight", 0.0, 100.0, 100.0 / max(len(selected_tickers), 1), 1.0)
        raw_sum = sum(custom_raw.values())
        custom_weights = {k: (v / raw_sum if raw_sum > 0 else 0) for k, v in custom_raw.items()}

        st.divider()
        st.subheader("Risk Controls")
        rolling_window = st.slider("Rolling window", 21, 252, 63, 21)
        te_target = st.number_input("Tracking Error target", 0.0, 0.50, 0.06, 0.01)
        te_band = st.number_input("Tracking Error band", 0.0, 0.20, 0.02, 0.01)

        st.divider()
        st.subheader("Advanced Spread / Bollinger")
        boll_window = st.slider("Bollinger window", 21, 252, 63, 21)
        boll_std = st.slider("Bollinger standard deviations", 1.0, 3.0, 2.0, 0.25)

        st.divider()
        st.subheader("GARCH Controls")
        run_garch = st.checkbox("Run GARCH Lab", value=False)
        garch_mode = st.selectbox("GARCH model set", list(GARCH_MODEL_OPTIONS.keys()), index=0)

    if len(selected_tickers) < 2:
        st.error("Select at least two commodity instruments.")
        return
    if start_date < MIN_START_DATE:
        st.error("Start date cannot be earlier than 2018-01-01.")
        return
    if start_date >= end_date:
        st.error("Start date must be before end date.")
        return

    try:
        with st.spinner("Downloading Yahoo Finance data..."):
            raw_prices = download_yahoo_prices(tuple(selected_tickers), str(start_date), str(end_date))
            prices = prepare_price_matrix(raw_prices, min_valid_ratio, ffill_limit, allow_single=False)
            returns = prepare_returns(prices)

            raw_benchmark = download_yahoo_prices((BENCHMARK_TICKER,), str(start_date), str(end_date))
            benchmark_prices = prepare_price_matrix(raw_benchmark, min_valid_ratio, ffill_limit, allow_single=True)
            benchmark_returns = prepare_returns(benchmark_prices).iloc[:, 0].rename(BENCHMARK_TICKER)

            common = returns.index.intersection(benchmark_returns.index)
            returns = returns.loc[common]
            prices = prices.loc[common]
            benchmark_returns = benchmark_returns.loc[common]

    except Exception as exc:
        st.error(f"Data error: {exc}")
        st.info("Try ETF Proxy mode, fewer instruments, a longer date range, clear cache, or retry later if Yahoo throttles futures metadata.")
        return

    strategy_returns, strategy_weights, weights_df, opt_perf = build_strategy_set(prices, returns, rf, max_weight, custom_weights)

    # Align strategies to benchmark
    for name in list(strategy_returns.keys()):
        p, b = align_two(strategy_returns[name], benchmark_returns, name, "Benchmark")
        strategy_returns[name] = p
    benchmark_returns = benchmark_returns.reindex(next(iter(strategy_returns.values())).index).dropna()

    all_strategy_names = list(strategy_returns.keys())
    selected_strategies = st.sidebar.multiselect(
        "Strategies shown in charts",
        options=all_strategy_names,
        default=all_strategy_names,
    )
    if not selected_strategies:
        selected_strategies = all_strategy_names

    strategy_metrics = metrics_for_returns({k: strategy_returns[k] for k in selected_strategies}, benchmark_returns, rf)
    all_strategy_metrics = metrics_for_returns(strategy_returns, benchmark_returns, rf)
    interpretation = interpret_strategy_table(all_strategy_metrics)

    instrument_map = {f"{COMMODITY_UNIVERSE[t]['display']} ({t})": returns[t] for t in returns.columns}
    instrument_metrics = metrics_for_returns(instrument_map, benchmark_returns, rf)
    quality = data_quality_report(raw_prices, prices, returns)

    primary = selected_strategies[0]
    primary_row = all_strategy_metrics[all_strategy_metrics["Strategy / Instrument"] == primary].iloc[0]

    tabs = st.tabs([
        "Executive Dashboard",
        "Portfolio Weights",
        "Optimization",
        "Performance Metrics",
        "Risk Metrics",
        "Rolling Beta",
        "Tracking Error",
        "VaR / CVaR",
        "Drawdown",
        "Correlation",
        "Log Return Differences",
        "Rolling Sharpe",
        "GARCH Volatility Lab",
        "QuantStats",
        "Info Hub",
        "Data Quality",
        "Export Center",
    ])

    with tabs[0]:
        st.subheader("Advanced Institutional KPI Layout")
        mode_label = "ETF Proxy" if "ETF Proxies" in universe_mode else "Futures"
        st.markdown(
            f"""
            <div class="qfa-note">
            Primary strategy: <b>{primary}</b>. Active data mode: <b>{mode_label}</b>.
            Sidebar parameters remain active: custom weights, RF rate, max weight constraint, rolling window, TE bands, GARCH model set and Bollinger settings.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div class='qfa-kpi-band'><div class='qfa-kpi-band-title'>Portfolio Risk / Return Command Center</div>", unsafe_allow_html=True)
        c = st.columns(8)
        with c[0]: kpi_card("Annual Return", fmt_pct(primary_row["Annual Return"]), primary, tone_for_return(primary_row["Annual Return"]))
        with c[1]: kpi_card("Annual Volatility", fmt_pct(primary_row["Annual Volatility"]), "risk level", "neutral")
        with c[2]: kpi_card("Sharpe", fmt_num(primary_row["Sharpe"]), f"RF {rf:.2%}", tone_for_sharpe(primary_row["Sharpe"]))
        with c[3]: kpi_card("Max Drawdown", fmt_pct(primary_row["Max Drawdown"]), "peak-to-trough", tone_for_drawdown(primary_row["Max Drawdown"]))
        with c[4]: kpi_card("VaR 95%", fmt_pct(primary_row["VaR 95% Daily"]), "daily", "warn")
        with c[5]: kpi_card("CVaR 95%", fmt_pct(primary_row["CVaR 95% Daily"]), "tail loss", "bad")
        with c[6]: kpi_card("Tracking Error", fmt_pct(primary_row["Tracking Error vs Benchmark"]), "vs ^GSPC", tone_for_te(primary_row["Tracking Error vs Benchmark"]))
        with c[7]: kpi_card("Beta", fmt_num(primary_row["Beta vs Benchmark"]), "vs ^GSPC", "neutral")

        st.markdown("</div>", unsafe_allow_html=True)

        c2 = st.columns(5)
        with c2[0]: kpi_card("Data Mode", mode_label, "transparent source", "neutral")
        with c2[1]: kpi_card("Instruments", str(len(selected_tickers)), "selected", "neutral")
        with c2[2]: kpi_card("Obs.", str(len(returns)), "daily aligned", "neutral")
        with c2[3]: kpi_card("Max Weight", fmt_pct(max_weight), "optimization cap", "neutral")
        with c2[4]: kpi_card("Rolling Window", str(rolling_window), "trading days", "neutral")

        safe_chart(fig_strategy_cumulative(strategy_returns, benchmark_returns, selected_strategies))
        safe_chart(fig_risk_return(strategy_metrics))
        safe_df(interpretation)

    with tabs[1]:
        st.subheader("Portfolio Weights")
        st.markdown("<div class='qfa-note'>Weights are recalculated from sidebar inputs and optimization settings.</div>", unsafe_allow_html=True)
        safe_chart(fig_weights(weights_df, selected_strategies))
        safe_df(weights_df)

    with tabs[2]:
        st.subheader("Optimization")
        st.markdown("<div class='qfa-note'>PyPortfolioOpt computes Max Sharpe and Min Volatility with the selected max-weight constraint.</div>", unsafe_allow_html=True)
        safe_chart(fig_weights(weights_df, ["Max Sharpe", "Min Volatility"] if "Max Sharpe" in weights_df.columns else selected_strategies))
        safe_chart(fig_risk_return(all_strategy_metrics))
        safe_df(opt_perf)

    with tabs[3]:
        st.subheader("Performance Metrics")
        safe_chart(fig_performance_bars(strategy_metrics))
        safe_chart(fig_strategy_cumulative(strategy_returns, benchmark_returns, selected_strategies))
        safe_df(strategy_metrics)

    with tabs[4]:
        st.subheader("Risk Metrics")
        safe_chart(fig_te_beta_map(strategy_metrics))
        safe_chart(fig_tail_bars(strategy_metrics))
        safe_df(strategy_metrics)
        st.subheader("User-Friendly Metric Dictionary")
        safe_df(metric_dictionary())

    with tabs[5]:
        st.subheader("Rolling Beta")
        beta_fig = make_subplots(rows=1, cols=1)
        for name in selected_strategies:
            beta = rolling_beta(strategy_returns[name], benchmark_returns, rolling_window)
            beta_fig.add_trace(go.Scatter(x=beta.index, y=beta.values, mode="lines", name=name))
        beta_fig.add_hline(y=1.0, line_dash="dash", annotation_text="Beta = 1")
        safe_chart(layout(beta_fig, f"{rolling_window}-Day Rolling Beta vs {BENCHMARK_NAME}", 720))

    with tabs[6]:
        st.subheader("Tracking Error")
        te_fig = make_subplots(rows=1, cols=1)
        for name in selected_strategies:
            te = rolling_tracking_error(strategy_returns[name], benchmark_returns, rolling_window)
            te_fig.add_trace(go.Scatter(x=te.index, y=te.values, mode="lines", name=name))
        te_fig.add_hline(y=te_target, line_dash="dash", annotation_text="TE Target")
        te_fig.add_hline(y=te_target + te_band, line_dash="dot", annotation_text="Upper Band")
        te_fig.add_hline(y=max(te_target - te_band, 0), line_dash="dot", annotation_text="Lower Band")
        te_fig.update_yaxes(tickformat=".0%")
        safe_chart(layout(te_fig, f"{rolling_window}-Day Rolling Tracking Error", 720))

    with tabs[7]:
        st.subheader("VaR / CVaR")
        safe_chart(fig_var_cvar(strategy_returns, selected_strategies, rolling_window))
        safe_chart(fig_tail_bars(strategy_metrics))

    with tabs[8]:
        st.subheader("Drawdown")
        safe_chart(fig_drawdowns(strategy_returns, benchmark_returns, selected_strategies))
        safe_chart(fig_underwater_recovery(strategy_returns, selected_strategies))

    with tabs[9]:
        st.subheader("Correlation")
        safe_chart(fig_corr(returns))
        st.subheader("Instrument Metrics")
        safe_df(instrument_metrics)

    with tabs[10]:
        st.subheader("ETF / Instrument Log Return Differences and Bollinger Bands")
        st.markdown(
            "<div class='qfa-note'>This tab shows log-return differences between selected instruments. In ETF Proxy mode, this makes proxy behavior transparent. Bollinger bands help identify unusual relative-return deviations.</div>",
            unsafe_allow_html=True,
        )
        lr = log_returns(prices)
        base_ticker = st.selectbox(
            "Base ticker for log-return differences",
            options=list(lr.columns),
            index=0,
            format_func=lambda t: COMMODITY_UNIVERSE.get(t, {}).get("display", t),
        )
        diff = return_difference_series(lr, base_ticker)
        safe_chart(fig_log_return_difference(lr, base_ticker))
        if not diff.empty:
            spread_choice = st.selectbox("Spread for Bollinger analysis", options=list(diff.columns), index=0)
            safe_chart(fig_log_return_difference_bollinger(diff, spread_choice, boll_window, boll_std))
            safe_df(diff.tail(250).reset_index().rename(columns={"index": "Date"}))
        else:
            st.info("No spread available. Select at least two instruments.")

    with tabs[11]:
        st.subheader("Rolling Sharpe")
        safe_chart(fig_rolling_sharpe(strategy_returns, selected_strategies, rf, rolling_window))
        st.markdown("<div class='qfa-note'>Rolling Sharpe helps users understand whether risk-adjusted performance is stable or only period-specific.</div>", unsafe_allow_html=True)

    with tabs[12]:
        st.subheader("GARCH Volatility Lab")
        st.markdown("<div class='qfa-note'>GARCH is fitted to individual commodities, not portfolio strategies.</div>", unsafe_allow_html=True)
        if run_garch:
            with st.spinner("Running GARCH models..."):
                garch_table, best_garch, garch_store = run_garch_suite(returns, tuple(GARCH_MODEL_OPTIONS[garch_mode]))
            safe_df(best_garch)
            safe_chart(fig_garch_best(best_garch, garch_store))
            safe_chart(fig_garch_rank(garch_table))

            if not best_garch.empty:
                choice = st.selectbox(
                    "Instrument diagnostics",
                    options=list(best_garch["Ticker"]),
                    format_func=lambda t: COMMODITY_UNIVERSE.get(t, {}).get("display", t),
                )
                best_key = best_garch[best_garch["Ticker"] == choice].iloc[0]["Best Model Key"]
                safe_chart(fig_garch_resid(choice, best_key, garch_store))
                with st.expander("ARCH summary"):
                    key = f"{choice}__{best_key}"
                    st.text(garch_store.get(key, {}).get("Summary", "Unavailable"))
            safe_df(garch_table)
        else:
            st.info("Enable 'Run GARCH Lab' in the sidebar.")

    with tabs[13]:
        st.subheader("QuantStats")
        st.markdown("<div class='qfa-note'>QuantStats report is generated for the selected strategy return stream.</div>", unsafe_allow_html=True)
        qs_strategy = st.selectbox("QuantStats strategy", options=all_strategy_names, index=all_strategy_names.index(primary) if primary in all_strategy_names else 0)
        qs_bytes = quantstats_html(qs_strategy, strategy_returns[qs_strategy], rf)
        if qs_bytes:
            st.download_button(
                f"Download QuantStats HTML — {qs_strategy}",
                data=qs_bytes,
                file_name=f"qfa_quantstats_{qs_strategy.lower().replace(' ', '_')}.html",
                mime="text/html",
            )
        else:
            st.warning("QuantStats unavailable or report generation failed.")

    with tabs[14]:
        st.subheader("Info Hub")
        st.subheader("Proxy Transparency")
        safe_df(PROXY_TRANSPARENCY_TABLE)
        st.caption("ETF proxy mode is explicit and visible. It is not synthetic data and not hidden futures replacement.")

        info_df = pd.DataFrame([
            {"Ticker": k, "Display Name": v["display"], "Full Name": v["name"], "Instrument Class": v["class"], "Source": "Yahoo Finance"}
            for k, v in COMMODITY_UNIVERSE.items()
        ])
        safe_df(info_df)
        st.subheader("Strategy Definitions")
        safe_df(pd.DataFrame([
            {"Strategy": "Equal Weight", "Definition": "Same weight for each selected instrument."},
            {"Strategy": "User Custom Weights", "Definition": "Sidebar weights normalized to 100%."},
            {"Strategy": "Inverse Volatility", "Definition": "Lower-volatility instruments receive larger weights."},
            {"Strategy": "Max Sharpe", "Definition": "PyPortfolioOpt expected Sharpe-maximizing portfolio."},
            {"Strategy": "Min Volatility", "Definition": "PyPortfolioOpt minimum-variance portfolio."},
        ]))
        st.subheader("Deployment Diagnostics")
        safe_df(pd.DataFrame([
            {"Item": "Version", "Value": VERSION},
            {"Item": "Python", "Value": sys.version.split()[0]},
            {"Item": "Streamlit", "Value": getattr(st, "__version__", "unknown")},
            {"Item": "Synthetic Fallback", "Value": "Disabled"},
            {"Item": "Benchmark", "Value": f"{BENCHMARK_NAME} ({BENCHMARK_TICKER})"},
        ]))

    with tabs[15]:
        st.subheader("Data Quality")
        safe_df(quality)
        st.caption(f"Aligned prices: {prices.shape}; aligned returns: {returns.shape}; benchmark returns: {benchmark_returns.shape}")

    with tabs[16]:
        st.subheader("Export Center")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button("Download strategy metrics CSV", df_csv(all_strategy_metrics), "qfa_strategy_metrics.csv", "text/csv")
            st.download_button("Download strategy weights CSV", df_csv(weights_df), "qfa_strategy_weights.csv", "text/csv")
        with col2:
            st.download_button("Download aligned prices CSV", prices.to_csv().encode("utf-8"), "qfa_aligned_prices.csv", "text/csv")
            st.download_button("Download aligned returns CSV", returns.to_csv().encode("utf-8"), "qfa_aligned_returns.csv", "text/csv")
        with col3:
            st.download_button("Download strategy returns CSV", returns_csv(strategy_returns), "qfa_strategy_returns.csv", "text/csv")
            st.download_button("Download data quality CSV", df_csv(quality), "qfa_data_quality.csv", "text/csv")


if __name__ == "__main__":
    main()
