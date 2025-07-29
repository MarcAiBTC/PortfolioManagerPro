"""
Utility functions for portfolio management.

This module encapsulates all helper routines required by the Streamlit
portfolio manager.  It handles persistence of user portfolios on disk,
retrieval of current and historical price data via yfinance, calculation
of a variety of performance and risk metrics, and generation of simple
recommendations.  The functions are designed to be self contained and
independent of Streamlit so they can be unit tested in isolation.

Key improvements over the initial version include:

* Robust caching of price data to minimise repeated network calls.
* Support for storing a “current” portfolio alongside historical
  timestamped versions.
* Comprehensive metric calculations (Alpha, Beta, RSI, Volatility,
  Sharpe ratio, value‑at‑risk).
* Simple diversification and rebalancing suggestions.
* Basic ticker validation and password strength checks.

Author: Enhanced by AI Assistant
"""

import os
import json
import logging
import time
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional, Iterable

import pandas as pd
import numpy as np

try:
    import yfinance as yf  # type: ignore
except Exception:
    yf = None

try:
    import plotly.express as px  # type: ignore
except Exception:
    px = None  # plotly is optional for certain visualisation helpers

# ---------------------------------------------------------------------------
# Configuration and caching constants
# ---------------------------------------------------------------------------

# Duration (in minutes) that cached current prices remain valid.  If
# updated prices are required more frequently this value can be reduced.
CACHE_DURATION_MINUTES: int = 5

# Determine availability of the yfinance package.  Many routines will
# gracefully degrade to returning NaNs when data cannot be fetched.
YF_AVAILABLE: bool = yf is not None

# In‑memory cache for current prices keyed by a comma‑separated list of
# tickers.  Each entry holds a mapping of ticker → price at the time of
# retrieval.
PRICE_CACHE: Dict[str, Dict[str, float]] = {}

# Timestamps of when each price cache entry was last updated.
CACHE_TIMESTAMPS: Dict[str, float] = {}

# Cache for historical price data.  Each key is a tuple (ticker, period)
# and the value is a pandas.Series of adjusted close prices.  This cache
# persists for the lifetime of the process.
HIST_PRICES_CACHE: Dict[Tuple[str, str], pd.Series] = {}

# Default risk‑free rate used for Sharpe ratio calculations.  For
# demonstration purposes we set this to zero; in a real world scenario
# this could be a short‑term Treasury yield.
RISK_FREE_RATE: float = 0.0

# Benchmark tickers used for Beta and Alpha calculations.  We use the
# S&P 500 (^GSPC) and NASDAQ 100 (^IXIC) by default.  Additional
# benchmarks can be added as needed.
DEFAULT_BENCHMARKS: List[str] = ["^GSPC", "^IXIC"]

# ---------------------------------------------------------------------------
# Filesystem paths
# ---------------------------------------------------------------------------

BASE_DIR = os.path.join(os.path.dirname(__file__), "user_data")
PORTFOLIO_DIR = os.path.join(BASE_DIR, "portfolios")

def _ensure_portfolio_dir() -> None:
    """Ensure the portfolio storage directory exists."""
    if not os.path.exists(PORTFOLIO_DIR):
        os.makedirs(PORTFOLIO_DIR, exist_ok=True)

def list_portfolios(username: str) -> List[str]:
    """
    List portfolio files associated with a given user.

    The returned list includes both timestamped historical files and a
    special file named ``<username>_current.<ext>`` which represents
    the most recently saved portfolio snapshot.  Files are ordered so
    that the newest (by modification time) appear first.

    Parameters
    ----------
    username : str
        The user's username.

    Returns
    -------
    List[str]
        Filenames of the user's portfolios sorted by modification time.
    """
    _ensure_portfolio_dir()
    files: List[str] = []
    for fname in os.listdir(PORTFOLIO_DIR):
        if not (fname.endswith(".csv") or fname.endswith(".json")):
            continue
        if fname.startswith(f"{username}_"):
            files.append(fname)
    files.sort(key=lambda f: os.path.getmtime(os.path.join(PORTFOLIO_DIR, f)), reverse=True)
    return files

def save_portfolio(
    username: str,
    df: pd.DataFrame,
    *,
    fmt: str = "csv",
    overwrite: bool = True,
    keep_history: bool = True,
) -> str:
    """
    Persist a user's portfolio to disk.

    Two variants of the save operation are supported:

    * When ``overwrite`` is True (the default), the portfolio is written
      to a file named ``<username>_current.<fmt>``.  This file always
      represents the most recent version of the portfolio and can be
      overwritten as often as needed.

    * When ``keep_history`` is True (also the default), a timestamped
      snapshot is created alongside the current file.  These snapshots
      enable users to revisit older versions of their portfolio.  If
      ``keep_history`` is False, no timestamped copy is created.

    Parameters
    ----------
    username : str
        User identifier.  Filenames are derived from this value.
    df : pandas.DataFrame
        Portfolio data.  It must contain at least the columns
        ``'Ticker'``, ``'Purchase Price'``, ``'Quantity'`` and
        ``'Asset Type'``.
    fmt : {'csv', 'json'}, optional
        File format to use.  Defaults to ``'csv'``.
    overwrite : bool, optional
        When True, overwrite the ``<username>_current.<fmt>`` file.
        When False, the current file is not touched.  Defaults to True.
    keep_history : bool, optional
        When True, write a timestamped snapshot in addition to the
        current file.  Defaults to True.

    Returns
    -------
    str
        The path to the most recently written file.  If both a
        ``current`` and a ``history`` file are written, this will be
        the path to the timestamped file.
    """
    _ensure_portfolio_dir()
    fmt = fmt.lower()
    if fmt not in {"csv", "json"}:
        raise ValueError("Unsupported format for portfolio saving")
    current_fname = f"{username}_current.{fmt}"
    current_path = os.path.join(PORTFOLIO_DIR, current_fname)
    if overwrite:
        if fmt == "csv":
            df.to_csv(current_path, index=False)
        else:
            df.to_json(current_path, orient="records", indent=2)
    written_path = current_path
    if keep_history:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_fname = f"{username}_{timestamp}.{fmt}"
        snapshot_path = os.path.join(PORTFOLIO_DIR, snapshot_fname)
        if fmt == "csv":
            df.to_csv(snapshot_path, index=False)
        else:
            df.to_json(snapshot_path, orient="records", indent=2)
        written_path = snapshot_path
    return written_path

def load_portfolio(username: str, filename: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Load a saved portfolio for a given user.

    If ``filename`` is provided, the specific file is loaded.  When
    ``filename`` is ``None``, the most recent portfolio (by
    modification time) is selected automatically.  If no portfolios are
    present for the user, ``None`` is returned.

    Parameters
    ----------
    username : str
        User identifier.
    filename : str, optional
        Specific filename to load.  Defaults to ``None``.

    Returns
    -------
    Optional[pandas.DataFrame]
        The loaded portfolio as a DataFrame, or ``None`` if the file
        cannot be read.
    """
    _ensure_portfolio_dir()
    files = list_portfolios(username)
    if not files:
        return None
    target = filename if filename is not None else files[0]
    fpath = os.path.join(PORTFOLIO_DIR, target)
    if not os.path.isfile(fpath):
        return None
    ext = os.path.splitext(fpath)[1].lower()
    try:
        if ext == ".csv":
            return pd.read_csv(fpath)
        else:
            return pd.read_json(fpath)
    except Exception:
        logging.exception("Failed to load portfolio %s", fpath)
        return None

# ---------------------------------------------------------------------------
# Price retrieval and caching
# ---------------------------------------------------------------------------

def fetch_current_prices(tickers: Iterable[str]) -> Dict[str, float]:
    """
    Retrieve current prices for a list of tickers using yfinance.

    If yfinance is not installed or network requests fail, the returned
    dictionary will have NaNs for missing values.

    Parameters
    ----------
    tickers : iterable of str
        The asset symbols to fetch.

    Returns
    -------
    Dict[str, float]
        Mapping from ticker to latest price.  Missing tickers map to np.nan.
    """
    tickers_list = [str(t).upper() for t in tickers]
    prices: Dict[str, float] = {t: np.nan for t in tickers_list}
    if not YF_AVAILABLE or not tickers_list:
        return prices
    try:
        data = yf.download(tickers=" ".join(tickers_list), period="1d", interval="1m", progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            for t in tickers_list:
                try:
                    prices[t] = float(data['Adj Close'][t].dropna().iloc[-1])
                except Exception:
                    prices[t] = np.nan
        else:
            try:
                prices[tickers_list[0]] = float(data['Adj Close'].dropna().iloc[-1])
            except Exception:
                prices[tickers_list[0]] = np.nan
    except Exception:
        for t in tickers_list:
            try:
                ticker_obj = yf.Ticker(t)
                info = ticker_obj.fast_info
                if info is None:
                    raise Exception
                prices[t] = float(info.get('lastPrice') or info.get('last_price') or info.get('previousClose') or np.nan)
            except Exception:
                prices[t] = np.nan
    return prices

def get_cached_prices(tickers: Iterable[str], cache_duration_minutes: int = CACHE_DURATION_MINUTES) -> Dict[str, float]:
    """
    Retrieve current prices with an intelligent in‑memory cache.

    A simple caching layer is implemented to avoid repeated calls to
    yfinance when the same set of tickers is requested within the
    ``cache_duration_minutes`` window.  When cached values are stale
    or absent the function falls back to ``fetch_current_prices``.

    Parameters
    ----------
    tickers : iterable of str
        Sequence of ticker symbols.
    cache_duration_minutes : int, optional
        Number of minutes cached prices remain valid.  Defaults to
        ``CACHE_DURATION_MINUTES``.

    Returns
    -------
    Dict[str, float]
        Mapping from ticker to the most recent price.  Missing or
        unsupported tickers map to ``numpy.nan``.
    """
    tickers_list = [str(t).upper() for t in tickers]
    if not tickers_list or not YF_AVAILABLE:
        return {ticker: np.nan for ticker in tickers_list}
    cache_key = ",".join(sorted(tickers_list))
    now = time.time()
    if cache_key in PRICE_CACHE:
        age = now - CACHE_TIMESTAMPS.get(cache_key, 0.0)
        if age < cache_duration_minutes * 60:
            return PRICE_CACHE[cache_key]
    prices = fetch_current_prices(tickers_list)
    PRICE_CACHE[cache_key] = prices
    CACHE_TIMESTAMPS[cache_key] = now
    return prices

# ---------------------------------------------------------------------------
# Basic metric calculations
# ---------------------------------------------------------------------------

def compute_metrics(df: pd.DataFrame, prices: Dict[str, float]) -> pd.DataFrame:
    """
    Compute portfolio metrics given purchase information and current prices.

    The returned DataFrame includes the following additional columns:

    - Current Price
    - Total Value (Current Price * Quantity)
    - Profit/Loss (absolute)
    - Profit/Loss (%)
    - Weight (%) (percentage of total portfolio value)

    Parameters
    ----------
    df : pandas.DataFrame
        The portfolio DataFrame with columns: 'Ticker', 'Purchase Price', 'Quantity', and 'Asset Type'.
    prices : dict
        Mapping from ticker to current price.

    Returns
    -------
    pandas.DataFrame
        DataFrame with computed metrics.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    required_cols = ['Ticker', 'Purchase Price', 'Quantity']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    df['Current Price'] = df['Ticker'].apply(lambda t: prices.get(str(t).upper(), np.nan))
    df['Total Value'] = df['Current Price'] * df['Quantity']
    df['Cost Basis'] = df['Purchase Price'] * df['Quantity']
    df['P/L'] = df['Total Value'] - df['Cost Basis']
    df['P/L %'] = np.where(df['Cost Basis'] > 0, (df['Total Value'] / df['Cost Basis'] - 1.0) * 100.0, np.nan)
    total_portfolio_value = df['Total Value'].sum()
    if total_portfolio_value > 0:
        df['Weight %'] = df['Total Value'] / total_portfolio_value * 100.0
    else:
        df['Weight %'] = np.nan
    return df

def compute_rsi(prices: pd.Series, period: int = 14) -> float:
    """
    Compute the Relative Strength Index (RSI) for a series of prices.

    RSI is a momentum oscillator that measures the magnitude of recent price
    changes to evaluate overbought or oversold conditions.

    Parameters
    ----------
    prices : pandas.Series
        Series of historical prices ordered by date.
    period : int, optional
        The lookback period for RSI.  Default is 14.

    Returns
    -------
    float
        The most recent RSI value.
    """
    if prices is None or len(prices) < period + 1:
        return float('nan')
    delta = prices.diff().dropna()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1.0/period, min_periods=period).mean()
    avg_loss = losses.ewm(alpha=1.0/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])

def compute_volatility(prices: pd.Series) -> float:
    """
    Compute the annualised volatility of a series of prices.

    Volatility is defined as the standard deviation of daily returns multiplied
    by the square root of the number of trading periods in a year (~252).

    Parameters
    ----------
    prices : pandas.Series
        Series of historical prices ordered by date.

    Returns
    -------
    float
        Annualised volatility as a percentage.
    """
    if prices is None or len(prices) < 2:
        return float('nan')
    returns = prices.pct_change().dropna()
    daily_vol = returns.std()
    annual_vol = daily_vol * np.sqrt(252)
    return float(annual_vol * 100.0)

def asset_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise the portfolio by asset type.

    Returns a DataFrame with columns 'Asset Type' and 'Total Value', showing the
    total value invested in each asset class.

    Parameters
    ----------
    df : pandas.DataFrame
        The portfolio DataFrame after computing metrics.  Must contain
        'Asset Type' and 'Total Value'.

    Returns
    -------
    pandas.DataFrame
        Summary DataFrame by asset class.
    """
    if df is None or df.empty or 'Asset Type' not in df.columns or 'Total Value' not in df.columns:
        return pd.DataFrame()
    summary = df.groupby('Asset Type')['Total Value'].sum().reset_index()
    summary = summary.sort_values('Total Value', ascending=False)
    return summary

def top_and_worst_assets(df: pd.DataFrame, n: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identify the top and worst performing assets based on percentage returns.

    Parameters
    ----------
    df : pandas.DataFrame
        Portfolio DataFrame with a 'P/L %' column.
    n : int, optional
        Number of assets to return in each category.  Default is 3.

    Returns
    -------
    tuple of (pandas.DataFrame, pandas.DataFrame)
        The first DataFrame contains the top performers, the second the worst performers.
    """
    if df is None or df.empty or 'P/L %' not in df.columns:
        return (pd.DataFrame(), pd.DataFrame())
    sorted_df = df.sort_values('P/L %', ascending=False)
    top_n = sorted_df.head(n)
    worst_n = sorted_df.tail(n).iloc[::-1]
    return top_n, worst_n

def suggest_diversification(df: pd.DataFrame) -> Optional[str]:
    """
    Provide a simple diversification suggestion based on asset allocation.

    If a single asset class represents more than 70% of the portfolio value,
    recommend diversification.

    Parameters
    ----------
    df : pandas.DataFrame
        Portfolio DataFrame with computed 'Weight %' and 'Asset Type' columns.

    Returns
    -------
    Optional[str]
        A suggestion string if diversification is needed; otherwise None.
    """
    if df is None or df.empty or 'Asset Type' not in df.columns or 'Weight %' not in df.columns:
        return None
    breakdown = df.groupby('Asset Type')['Weight %'].sum()
    max_type = breakdown.idxmax()
    max_weight = breakdown.max()
    if max_weight > 70:
        return (
            f"La mayor parte de tu cartera (≈{max_weight:.1f}%) está en {max_type}. "
            "Considera diversificar hacia otras clases de activos."
        )
    return None

# ---------------------------------------------------------------------------
# Historical data helpers and enhanced metrics
# ---------------------------------------------------------------------------

def _fetch_historical_series(ticker: str, period: str = "6mo", interval: str = "1d") -> pd.Series:
    """
    Internal helper to fetch a series of adjusted close prices for a single ticker.

    Results are cached for the duration of the process to minimise
    redundant network requests.  When yfinance is unavailable or an
    error occurs, an empty Series is returned.

    Parameters
    ----------
    ticker : str
        Symbol to fetch.
    period : str, optional
        Historical period (e.g. '1mo', '3mo', '6mo', '1y').  Defaults to '6mo'.
    interval : str, optional
        Data interval (e.g. '1d', '1wk').  Defaults to '1d'.

    Returns
    -------
    pandas.Series
        Series of adjusted close prices indexed by date.
    """
    key = (ticker.upper(), period)
    if key in HIST_PRICES_CACHE:
        return HIST_PRICES_CACHE[key]
    if not YF_AVAILABLE:
        HIST_PRICES_CACHE[key] = pd.Series(dtype=float)
        return HIST_PRICES_CACHE[key]
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        col = "Adj Close" if "Adj Close" in data.columns else "Close"
        series = data[col].dropna()
        HIST_PRICES_CACHE[key] = series
        return series
    except Exception:
        HIST_PRICES_CACHE[key] = pd.Series(dtype=float)
        return HIST_PRICES_CACHE[key]

def fetch_benchmark_data(period: str = "6mo", interval: str = "1d", tickers: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Retrieve historical price data for a set of benchmark indices.

    By default this function downloads the S&P 500 (\^GSPC) and
    NASDAQ 100 (\^IXIC) indices for the given period and interval.

    Parameters
    ----------
    period : str, optional
        History period (e.g. '6mo', '1y').  Defaults to '6mo'.
    interval : str, optional
        Data interval (e.g. '1d', '1wk').  Defaults to '1d'.
    tickers : list of str, optional
        Specific benchmark symbols to fetch.  If ``None``, uses
        ``DEFAULT_BENCHMARKS``.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by date with one column per benchmark
        containing the adjusted close prices.  Missing data are
        forward filled.
    """
    if tickers is None:
        tickers = DEFAULT_BENCHMARKS
    data: Dict[str, pd.Series] = {}
    for ticker in tickers:
        series = _fetch_historical_series(ticker, period=period, interval=interval)
        if not series.empty:
            data[ticker] = series
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df = df.sort_index().fillna(method="ffill")
    return df

def compute_enhanced_metrics(
    df: pd.DataFrame,
    prices: Dict[str, float],
    benchmark_data: Optional[pd.DataFrame] = None,
    period: str = "6mo",
) -> pd.DataFrame:
    """
    Compute an enriched set of metrics for each asset in a portfolio.

    This function builds upon ``compute_metrics`` by adding momentum
    (RSI), volatility, Alpha and Beta relative to a benchmark, and
    returns the result as a new DataFrame.  Missing values are filled
    with NaN.  If benchmark data is not provided or insufficient,
    Alpha and Beta will be NaN.

    Parameters
    ----------
    df : pandas.DataFrame
        Portfolio data with at least the columns ``Ticker``, ``Purchase Price``
        and ``Quantity``.
    prices : dict
        Mapping from ticker to current price.
    benchmark_data : pandas.DataFrame, optional
        Historical prices for benchmark indices (e.g. from
        ``fetch_benchmark_data``).  Must be indexed by date.  If
        ``None``, benchmark dependent metrics are skipped.
    period : str, optional
        The historical period to use when fetching individual asset
        prices.  Defaults to '6mo'.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing all original columns plus
        ``'Current Price'``, ``'Total Value'``, ``'Cost Basis'``, ``'P/L'``,
        ``'P/L %'``, ``'Weight %'``, ``'RSI'``, ``'Volatility'``,
        ``'Beta'`` and ``'Alpha'``.
    """
    base_metrics = compute_metrics(df, prices)
    if base_metrics.empty:
        return base_metrics
    base_metrics = base_metrics.copy()
    base_metrics["RSI"] = np.nan
    base_metrics["Volatility"] = np.nan
    base_metrics["Beta"] = np.nan
    base_metrics["Alpha"] = np.nan
    benchmark_returns: Optional[pd.Series] = None
    if benchmark_data is not None and not benchmark_data.empty:
        primary_bench = benchmark_data.iloc[:, 0].dropna()
        benchmark_returns = primary_bench.pct_change().dropna()
    for idx, row in base_metrics.iterrows():
        ticker = str(row.get("Ticker", "")).upper()
        if not ticker:
            continue
        hist_prices = _fetch_historical_series(ticker, period=period)
        if not hist_prices.empty:
            rsi = compute_rsi(hist_prices)
            base_metrics.at[idx, "RSI"] = rsi
            vol = compute_volatility(hist_prices)
            base_metrics.at[idx, "Volatility"] = vol
            if benchmark_returns is not None and len(hist_prices) > 1:
                asset_returns = hist_prices.pct_change().dropna()
                joined = pd.concat([asset_returns, benchmark_returns], axis=1, join="inner").dropna()
                if not joined.empty and joined.shape[0] > 1:
                    cov = np.cov(joined.iloc[:, 0], joined.iloc[:, 1])[0, 1]
                    var = np.var(joined.iloc[:, 1])
                    beta = cov / var if var != 0 else np.nan
                    base_metrics.at[idx, "Beta"] = beta
                    avg_asset = joined.iloc[:, 0].mean()
                    avg_bench = joined.iloc[:, 1].mean()
                    expected_return = beta * avg_bench if beta is not np.nan else np.nan
                    alpha = avg_asset - expected_return if expected_return is not np.nan else np.nan
                    base_metrics.at[idx, "Alpha"] = alpha
    return base_metrics

# ---------------------------------------------------------------------------
# Portfolio level risk measures
# ---------------------------------------------------------------------------

def _download_multiple_series(tickers: List[str], period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    """
    Download historical adjusted close prices for multiple tickers at once.

    This helper utilises yfinance's ability to fetch several tickers in a
    single request, reducing network overhead.  If yfinance is
    unavailable, an empty DataFrame is returned.

    Parameters
    ----------
    tickers : list of str
        List of symbols to fetch.
    period : str, optional
        History period.  Defaults to '6mo'.
    interval : str, optional
        Data interval.  Defaults to '1d'.

    Returns
    -------
    pandas.DataFrame
        DataFrame of adjusted close prices indexed by date.
    """
    if not YF_AVAILABLE or not tickers:
        return pd.DataFrame()
    try:
        data = yf.download(tickers=" ".join(tickers), period=period, interval=interval, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            if "Adj Close" in data.columns.get_level_values(0):
                adj = data["Adj Close"]
            else:
                adj = data["Close"]
        else:
            col = "Adj Close" if "Adj Close" in data.columns else "Close"
            adj = data[[col]]
            adj.columns = [tickers[0]]
        adj = adj.dropna(how="all")
        return adj
    except Exception:
        return pd.DataFrame()

def calculate_portfolio_sharpe(
    metrics_df: pd.DataFrame,
    period: str = "6mo",
    risk_free_rate: float = RISK_FREE_RATE,
) -> float:
    """
    Compute the annualised Sharpe ratio for the entire portfolio.

    The Sharpe ratio is defined as the portfolio's excess return over
    the risk‑free rate divided by the standard deviation of portfolio
    returns.  Daily returns for each asset are weighted by the
    portfolio weights (from the ``'Weight %'`` column).  The resulting
    series is annualised by multiplying by √252.

    Parameters
    ----------
    metrics_df : pandas.DataFrame
        DataFrame with computed metrics including ``'Ticker'`` and
        ``'Weight %'`` columns.
    period : str, optional
        Period to use for historical returns.  Defaults to '6mo'.
    risk_free_rate : float, optional
        Daily risk‑free rate to subtract from returns.  Defaults to 0.0.

    Returns
    -------
    float
        Annualised Sharpe ratio.  Returns NaN if it cannot be
        computed.
    """
    if metrics_df is None or metrics_df.empty or "Ticker" not in metrics_df.columns or "Weight %" not in metrics_df.columns:
        return float("nan")
    weights = metrics_df["Weight %"].astype(float) / 100.0
    tickers = metrics_df["Ticker"].astype(str).str.upper().tolist()
    price_df = _download_multiple_series(tickers, period=period)
    if price_df.empty:
        return float("nan")
    returns = price_df.pct_change().dropna()
    try:
        returns = returns[tickers]
    except KeyError:
        common = [t for t in tickers if t in returns.columns]
        if not common:
            return float("nan")
        returns = returns[common]
        mask = metrics_df["Ticker"].isin(common)
        weights = weights[mask]
        weights = weights / weights.sum()
    portfolio_returns = (returns * weights.values).sum(axis=1)
    excess_returns = portfolio_returns - risk_free_rate
    mean_excess = excess_returns.mean()
    std_excess = excess_returns.std()
    if std_excess == 0 or np.isnan(std_excess):
        return float("nan")
    sharpe_daily = mean_excess / std_excess
    sharpe_annual = sharpe_daily * np.sqrt(252)
    return float(sharpe_annual)

def calculate_value_at_risk(
    metrics_df: pd.DataFrame,
    confidence: float = 0.95,
    period: str = "6mo",
) -> float:
    """
    Estimate the one‑day Value at Risk (VaR) for the portfolio.

    VaR is calculated by constructing the distribution of daily
    portfolio returns over the specified period and taking the
    appropriate lower percentile.  The returned value represents the
    potential loss (in dollars) that will not be exceeded with the
    given confidence level.

    Parameters
    ----------
    metrics_df : pandas.DataFrame
        DataFrame with computed metrics including ``'Ticker'``,
        ``'Weight %'`` and ``'Total Value'`` columns.
    confidence : float, optional
        Confidence level (between 0 and 1).  Defaults to 0.95 for a
        95% VaR.
    period : str, optional
        Historical period to use for VaR calculation.  Defaults to '6mo'.

    Returns
    -------
    float
        Estimated dollar VaR.  Returns NaN if it cannot be computed.
    """
    if metrics_df is None or metrics_df.empty or "Ticker" not in metrics_df.columns or "Weight %" not in metrics_df.columns or "Total Value" not in metrics_df.columns:
        return float("nan")
    weights = metrics_df["Weight %"].astype(float) / 100.0
    tickers = metrics_df["Ticker"].astype(str).str.upper().tolist()
    price_df = _download_multiple_series(tickers, period=period)
    if price_df.empty:
        return float("nan")
    returns = price_df.pct_change().dropna()
    try:
        returns = returns[tickers]
    except KeyError:
        common = [t for t in tickers if t in returns.columns]
        if not common:
            return float("nan")
        returns = returns[common]
        mask = metrics_df["Ticker"].isin(common)
        weights = weights[mask]
        weights = weights / weights.sum()
    portfolio_returns = (returns * weights.values).sum(axis=1)
    losses = -portfolio_returns
    var_threshold = np.nanpercentile(losses, confidence * 100)
    total_value = metrics_df["Total Value"].sum()
    return float(var_threshold * total_value)

# ---------------------------------------------------------------------------
# Recommendations and suggestions
# ---------------------------------------------------------------------------

def generate_portfolio_recommendations(metrics_df: pd.DataFrame) -> List[Dict[str, str]]:
    """
    Produce a list of simple recommendations based on portfolio metrics.

    Each recommendation is a dict containing a ``title``, ``description``
    and ``type`` (one of 'warning', 'success' or 'info').

    Parameters
    ----------
    metrics_df : pandas.DataFrame
        DataFrame with enhanced metrics computed via
        ``compute_enhanced_metrics``.

    Returns
    -------
    List[Dict[str, str]]
        A list of recommendation objects.  The list may be empty.
    """
    if metrics_df is None or metrics_df.empty:
        return []
    recs: List[Dict[str, str]] = []
    divers_msg = suggest_diversification(metrics_df)
    if divers_msg:
        recs.append({
            "type": "warning",
            "title": "Diversificación",
            "description": divers_msg,
        })
    if "Volatility" in metrics_df.columns:
        high_vol = metrics_df[metrics_df["Volatility"] > 40]
        if not high_vol.empty:
            tickers_list = ", ".join(high_vol["Ticker"].astype(str).tolist())
            recs.append({
                "type": "warning",
                "title": "Alta volatilidad",
                "description": f"Los siguientes activos presentan volatilidad elevada (>40% anual): {tickers_list}. Considera reducir exposición o equilibrar con activos más estables.",
            })
    if "RSI" in metrics_df.columns:
        oversold = metrics_df[metrics_df["RSI"] < 30]
        overbought = metrics_df[metrics_df["RSI"] > 70]
        if not oversold.empty:
            recs.append({
                "type": "info",
                "title": "RSI bajo (posible oportunidad)",
                "description": f"Algunos activos tienen RSI <30 y podrían estar sobrevendidos: {', '.join(oversold['Ticker'].astype(str))}. Estudia si es buen momento para comprar.",
            })
        if not overbought.empty:
            recs.append({
                "type": "info",
                "title": "RSI alto (posible sobrecompra)",
                "description": f"Algunos activos tienen RSI >70 y podrían estar sobrecomprados: {', '.join(overbought['Ticker'].astype(str))}. Vigila posibles correcciones.",
            })
    if "P/L %" in metrics_df.columns:
        big_losses = metrics_df[metrics_df["P/L %"] < -20]
        if not big_losses.empty:
            recs.append({
                "type": "warning",
                "title": "Pérdidas significativas",
                "description": f"{', '.join(big_losses['Ticker'].astype(str))} acumulan pérdidas superiores al 20%. Evalúa si mantener o deshacer la posición.",
            })
    if not recs:
        recs.append({
            "type": "success",
            "title": "Buen rendimiento",
            "description": "Tu cartera está bien diversificada y no presenta alertas destacables.",
        })
    return recs

def suggest_rebalancing(metrics_df: pd.DataFrame) -> Optional[Dict[str, pd.DataFrame]]:
    """
    Suggest a target asset class allocation to encourage diversification.

    This function compares the current portfolio allocation by asset
    type against a simple target allocation.  The target weights are
    arbitrary and can be adjusted to suit individual risk preferences.

    Parameters
    ----------
    metrics_df : pandas.DataFrame
        DataFrame with a ``'Asset Type'`` and ``'Weight %'`` columns.

    Returns
    -------
    Optional[Dict[str, pandas.DataFrame]]
        A dict with two keys, ``'current'`` and ``'suggested'``, each
        mapping to a DataFrame with columns ``'asset_type'`` and
        ``'weight'``.  Returns ``None`` if inputs are invalid.
    """
    if metrics_df is None or metrics_df.empty or "Asset Type" not in metrics_df.columns or "Weight %" not in metrics_df.columns:
        return None
    current = metrics_df.groupby("Asset Type")["Weight %"].sum().reset_index()
    current.columns = ["asset_type", "weight"]
    unique_types = set(current["asset_type"].tolist())
    target_weights: Dict[str, float] = {}
    for atype in unique_types:
        key = atype.lower()
        if key in {"stock", "acciones"}:
            target_weights[atype] = 50.0
        elif key in {"etf"}:
            target_weights[atype] = 20.0
        elif key in {"bond", "bono", "bonos"}:
            target_weights[atype] = 10.0
        elif key in {"crypto", "criptomoneda"}:
            target_weights[atype] = 5.0
        elif key in {"commodity", "oro", "plata", "materias primas"}:
            target_weights[atype] = 5.0
        else:
            target_weights[atype] = 10.0 / max(len(unique_types), 1)
    total_target = sum(target_weights.values())
    for k in target_weights:
        target_weights[k] = target_weights[k] / total_target * 100.0
    suggested = pd.DataFrame({
        "asset_type": list(target_weights.keys()),
        "weight": list(target_weights.values()),
    })
    return {
        "current": current,
        "suggested": suggested
    }

# ---------------------------------------------------------------------------
# Validation and password strength helpers
# ---------------------------------------------------------------------------

def validate_tickers(tickers: Iterable[str]) -> Dict[str, bool]:
    """
    Validate a list of ticker symbols using yfinance.

    This function attempts to fetch a minimal amount of data for each
    symbol to determine if it is valid.  Invalid symbols will return
    ``False``.  If yfinance is unavailable all tickers are marked
    invalid.

    Parameters
    ----------
    tickers : iterable of str
        Ticker symbols to validate.

    Returns
    -------
    Dict[str, bool]
        Mapping of ticker symbol to a boolean indicating validity.
    """
    results: Dict[str, bool] = {}
    if not YF_AVAILABLE:
        for t in tickers:
            results[str(t).upper()] = False
        return results
    for ticker in tickers:
        sym = str(ticker).upper().strip()
        if not sym:
            results[sym] = False
            continue
        try:
            data = yf.download(sym, period="1d", interval="1d", progress=False)
            results[sym] = not data.empty
        except Exception:
            results[sym] = False
    return results

def check_password_strength(password: str) -> str:
    """
    Assess the strength of a password using simple heuristics.

    The algorithm assigns a score based on length and character
    diversity (uppercase, lowercase, digits, symbols).  The result is
    classified into ``'Weak'``, ``'Medium'`` or ``'Strong'``.

    Parameters
    ----------
    password : str
        Password to evaluate.

    Returns
    -------
    str
        One of 'Weak', 'Medium' or 'Strong'.
    """
    score = 0
    length = len(password or "")
    if length >= 8:
        score += 1
    if any(c.islower() for c in password):
        score += 1
    if any(c.isupper() for c in password):
        score += 1
    if any(c.isdigit() for c in password):
        score += 1
    if any(not c.isalnum() for c in password):
        score += 1
    if score <= 2:
        return "Weak"
    elif score == 3:
        return "Medium"
    else:
        return "Strong"