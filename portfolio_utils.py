"""
Enhanced Portfolio Utilities Module
===================================

This module provides comprehensive portfolio management utilities including:
- Robust data fetching with intelligent caching
- Advanced financial metrics calculation
- Portfolio analysis and recommendations
- File I/O operations with error handling
- Data validation and cleaning

Key improvements:
- Fixed deprecated pandas methods
- Better error handling and logging
- Enhanced caching mechanisms
- More robust price fetching
- Comprehensive metric calculations

Author: Enhanced by AI Assistant
"""

import os
import json
import logging
import time
import warnings
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional, Iterable, Union

import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress pandas warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Try to import yfinance
try:
    import yfinance as yf
    YF_AVAILABLE = True
    logger.info("yfinance successfully imported")
except ImportError as e:
    yf = None
    YF_AVAILABLE = False
    logger.warning(f"yfinance not available: {e}")

# Try to import plotly for visualization helpers
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    px = None
    PLOTLY_AVAILABLE = False

# ============================================================================
# Configuration Constants
# ============================================================================

# Cache duration for price data (in minutes)
CACHE_DURATION_MINUTES: int = 5

# Risk-free rate for Sharpe ratio calculations (as decimal)
RISK_FREE_RATE: float = 0.02  # 2% annual

# Default benchmark tickers
DEFAULT_BENCHMARKS: List[str] = ["^GSPC", "^IXIC"]  # S&P 500, NASDAQ

# Maximum number of tickers to fetch at once
MAX_BATCH_SIZE: int = 50

# ============================================================================
# Global Cache Variables
# ============================================================================

# In-memory cache for current prices
PRICE_CACHE: Dict[str, Dict[str, float]] = {}
CACHE_TIMESTAMPS: Dict[str, float] = {}

# Cache for historical price data
HIST_PRICES_CACHE: Dict[Tuple[str, str], pd.Series] = {}

# Cache for benchmark data
BENCHMARK_CACHE: Dict[str, pd.DataFrame] = {}

# ============================================================================
# File System Setup
# ============================================================================

BASE_DIR = os.path.join(os.path.dirname(__file__), "user_data")
PORTFOLIO_DIR = os.path.join(BASE_DIR, "portfolios")

def _ensure_portfolio_dir() -> None:
    """Ensure the portfolio storage directory exists."""
    try:
        os.makedirs(PORTFOLIO_DIR, exist_ok=True)
        logger.debug(f"Portfolio directory ensured: {PORTFOLIO_DIR}")
    except Exception as e:
        logger.error(f"Failed to create portfolio directory: {e}")
        raise

# ============================================================================
# Portfolio File Operations
# ============================================================================

def list_portfolios(username: str) -> List[str]:
    """
    List portfolio files for a given user, sorted by modification time.
    
    Parameters
    ----------
    username : str
        The user's username
        
    Returns
    -------
    List[str]
        List of portfolio filenames sorted by modification time (newest first)
    """
    _ensure_portfolio_dir()
    files = []
    
    try:
        for fname in os.listdir(PORTFOLIO_DIR):
            if not (fname.endswith(".csv") or fname.endswith(".json")):
                continue
            if fname.startswith(f"{username}_"):
                files.append(fname)
        
        # Sort by modification time (newest first)
        files.sort(
            key=lambda f: os.path.getmtime(os.path.join(PORTFOLIO_DIR, f)), 
            reverse=True
        )
        
        logger.debug(f"Found {len(files)} portfolios for user {username}")
        return files
        
    except Exception as e:
        logger.error(f"Error listing portfolios for {username}: {e}")
        return []

def save_portfolio(
    username: str,
    df: pd.DataFrame,
    *,
    fmt: str = "csv",
    overwrite: bool = True,
    keep_history: bool = True,
) -> str:
    """
    Save a portfolio to disk with optional history keeping.
    
    Parameters
    ----------
    username : str
        User identifier
    df : pd.DataFrame
        Portfolio data to save
    fmt : str, default "csv"
        File format ("csv" or "json")
    overwrite : bool, default True
        Whether to overwrite the current file
    keep_history : bool, default True
        Whether to keep a timestamped copy
        
    Returns
    -------
    str
        Path to the saved file
    """
    _ensure_portfolio_dir()
    
    if fmt not in {"csv", "json"}:
        raise ValueError(f"Unsupported format: {fmt}")
    
    try:
        # Save current file
        current_fname = f"{username}_current.{fmt}"
        current_path = os.path.join(PORTFOLIO_DIR, current_fname)
        
        if overwrite:
            if fmt == "csv":
                df.to_csv(current_path, index=False)
            else:
                df.to_json(current_path, orient="records", indent=2)
            logger.info(f"Saved current portfolio for {username}: {current_path}")
        
        written_path = current_path
        
        # Save timestamped copy if requested
        if keep_history:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot_fname = f"{username}_{timestamp}.{fmt}"
            snapshot_path = os.path.join(PORTFOLIO_DIR, snapshot_fname)
            
            if fmt == "csv":
                df.to_csv(snapshot_path, index=False)
            else:
                df.to_json(snapshot_path, orient="records", indent=2)
            
            written_path = snapshot_path
            logger.info(f"Saved portfolio snapshot: {snapshot_path}")
        
        return written_path
        
    except Exception as e:
        logger.error(f"Error saving portfolio for {username}: {e}")
        raise

def load_portfolio(username: str, filename: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Load a portfolio from disk.
    
    Parameters
    ----------
    username : str
        User identifier
    filename : str, optional
        Specific filename to load. If None, loads most recent.
        
    Returns
    -------
    Optional[pd.DataFrame]
        Loaded portfolio data or None if failed
    """
    _ensure_portfolio_dir()
    
    try:
        files = list_portfolios(username)
        if not files:
            logger.info(f"No portfolios found for user {username}")
            return None
        
        target = filename if filename is not None else files[0]
        fpath = os.path.join(PORTFOLIO_DIR, target)
        
        if not os.path.isfile(fpath):
            logger.error(f"Portfolio file not found: {fpath}")
            return None
        
        ext = os.path.splitext(fpath)[1].lower()
        
        if ext == ".csv":
            df = pd.read_csv(fpath)
        elif ext == ".json":
            df = pd.read_json(fpath)
        else:
            logger.error(f"Unsupported file format: {ext}")
            return None
        
        logger.info(f"Loaded portfolio for {username}: {len(df)} assets from {target}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading portfolio for {username}: {e}")
        return None

# ============================================================================
# Price Fetching and Caching
# ============================================================================

def fetch_current_prices(tickers: Iterable[str]) -> Dict[str, float]:
    """
    Fetch current prices for a list of tickers using yfinance.
    
    Parameters
    ----------
    tickers : Iterable[str]
        Asset symbols to fetch
        
    Returns
    -------
    Dict[str, float]
        Mapping from ticker to current price (NaN for failures)
    """
    if not YF_AVAILABLE:
        logger.warning("yfinance not available - returning NaN prices")
        return {str(t).upper(): np.nan for t in tickers}
    
    tickers_list = [str(t).upper().strip() for t in tickers if str(t).strip()]
    prices = {t: np.nan for t in tickers_list}
    
    if not tickers_list:
        return prices
    
    try:
        # Batch fetch for efficiency
        tickers_str = " ".join(tickers_list)
        logger.debug(f"Fetching prices for: {tickers_str}")
        
        # Try downloading recent data
        data = yf.download(
            tickers=tickers_str, 
            period="1d", 
            interval="1m", 
            progress=False,
            show_errors=False
        )
        
        if data.empty:
            logger.warning("No data returned from yfinance download")
            return prices
        
        # Handle multi-index columns (multiple tickers)
        if isinstance(data.columns, pd.MultiIndex):
            for ticker in tickers_list:
                try:
                    if ('Adj Close', ticker) in data.columns:
                        price_series = data[('Adj Close', ticker)].dropna()
                    elif ('Close', ticker) in data.columns:
                        price_series = data[('Close', ticker)].dropna()
                    else:
                        continue
                    
                    if not price_series.empty:
                        prices[ticker] = float(price_series.iloc[-1])
                        
                except Exception as e:
                    logger.debug(f"Error getting price for {ticker}: {e}")
                    continue
        else:
            # Single ticker
            if len(tickers_list) == 1:
                ticker = tickers_list[0]
                try:
                    if 'Adj Close' in data.columns:
                        price_series = data['Adj Close'].dropna()
                    elif 'Close' in data.columns:
                        price_series = data['Close'].dropna()
                    else:
                        return prices
                    
                    if not price_series.empty:
                        prices[ticker] = float(price_series.iloc[-1])
                        
                except Exception as e:
                    logger.debug(f"Error getting price for {ticker}: {e}")
        
        # Fill in any missing prices with individual ticker requests
        missing_tickers = [t for t, p in prices.items() if pd.isna(p)]
        for ticker in missing_tickers:
            try:
                ticker_obj = yf.Ticker(ticker)
                info = ticker_obj.fast_info
                
                if hasattr(info, 'last_price') and info.last_price:
                    prices[ticker] = float(info.last_price)
                elif hasattr(info, 'previousClose') and info.previousClose:
                    prices[ticker] = float(info.previousClose)
                    
            except Exception as e:
                logger.debug(f"Failed to get individual price for {ticker}: {e}")
                continue
        
        successful_prices = sum(1 for p in prices.values() if not pd.isna(p))
        logger.info(f"Successfully fetched {successful_prices}/{len(tickers_list)} prices")
        
        return prices
        
    except Exception as e:
        logger.error(f"Error fetching current prices: {e}")
        return prices

def get_cached_prices(
    tickers: Iterable[str], 
    cache_duration_minutes: int = CACHE_DURATION_MINUTES
) -> Dict[str, float]:
    """
    Get current prices with intelligent caching.
    
    Parameters
    ----------
    tickers : Iterable[str]
        Asset symbols to fetch
    cache_duration_minutes : int, default CACHE_DURATION_MINUTES
        Cache validity duration in minutes
        
    Returns
    -------
    Dict[str, float]
        Mapping from ticker to current price
    """
    tickers_list = [str(t).upper().strip() for t in tickers if str(t).strip()]
    
    if not tickers_list or not YF_AVAILABLE:
        return {ticker: np.nan for ticker in tickers_list}
    
    # Create cache key
    cache_key = ",".join(sorted(tickers_list))
    now = time.time()
    
    # Check if we have valid cached data
    if cache_key in PRICE_CACHE:
        cache_age = now - CACHE_TIMESTAMPS.get(cache_key, 0.0)
        if cache_age < cache_duration_minutes * 60:
            logger.debug(f"Using cached prices for {len(tickers_list)} tickers")
            return PRICE_CACHE[cache_key]
    
    # Fetch fresh data
    logger.debug(f"Fetching fresh prices for {len(tickers_list)} tickers")
    prices = fetch_current_prices(tickers_list)
    
    # Update cache
    PRICE_CACHE[cache_key] = prices
    CACHE_TIMESTAMPS[cache_key] = now
    
    return prices

# ============================================================================
# Historical Data and Advanced Metrics
# ============================================================================

def fetch_historical_series(
    ticker: str, 
    period: str = "6mo", 
    interval: str = "1d"
) -> pd.Series:
    """
    Fetch historical price series for a single ticker with caching.
    
    Parameters
    ----------
    ticker : str
        Asset symbol
    period : str, default "6mo"
        Historical period
    interval : str, default "1d"
        Data interval
        
    Returns
    -------
    pd.Series
        Historical adjusted close prices
    """
    if not YF_AVAILABLE:
        return pd.Series(dtype=float)
    
    cache_key = (ticker.upper(), period)
    
    # Check cache
    if cache_key in HIST_PRICES_CACHE:
        return HIST_PRICES_CACHE[cache_key]
    
    try:
        logger.debug(f"Fetching historical data for {ticker} ({period})")
        data = yf.download(
            ticker, 
            period=period, 
            interval=interval, 
            progress=False,
            show_errors=False
        )
        
        if data.empty:
            series = pd.Series(dtype=float)
        else:
            # Use Adj Close if available, otherwise Close
            if 'Adj Close' in data.columns:
                series = data['Adj Close'].dropna()
            elif 'Close' in data.columns:
                series = data['Close'].dropna()
            else:
                series = pd.Series(dtype=float)
        
        # Cache the result
        HIST_PRICES_CACHE[cache_key] = series
        logger.debug(f"Cached {len(series)} historical prices for {ticker}")
        
        return series
        
    except Exception as e:
        logger.error(f"Error fetching historical data for {ticker}: {e}")
        empty_series = pd.Series(dtype=float)
        HIST_PRICES_CACHE[cache_key] = empty_series
        return empty_series

def fetch_benchmark_data(
    period: str = "6mo", 
    interval: str = "1d", 
    tickers: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Fetch benchmark index data with caching.
    
    Parameters
    ----------
    period : str, default "6mo"
        Historical period
    interval : str, default "1d"
        Data interval
    tickers : List[str], optional
        Benchmark tickers to fetch
        
    Returns
    -------
    pd.DataFrame
        Benchmark price data indexed by date
    """
    if not YF_AVAILABLE:
        return pd.DataFrame()
    
    if tickers is None:
        tickers = DEFAULT_BENCHMARKS
    
    cache_key = f"{','.join(tickers)}_{period}"
    
    # Check cache
    if cache_key in BENCHMARK_CACHE:
        return BENCHMARK_CACHE[cache_key]
    
    try:
        logger.debug(f"Fetching benchmark data: {tickers} ({period})")
        data = {}
        
        for ticker in tickers:
            series = fetch_historical_series(ticker, period=period, interval=interval)
            if not series.empty:
                data[ticker] = series
        
        if not data:
            df = pd.DataFrame()
        else:
            df = pd.DataFrame(data).sort_index()
            # Forward fill missing values
            df = df.ffill()
        
        # Cache the result
        BENCHMARK_CACHE[cache_key] = df
        logger.debug(f"Cached benchmark data: {df.shape}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error fetching benchmark data: {e}")
        empty_df = pd.DataFrame()
        BENCHMARK_CACHE[cache_key] = empty_df
        return empty_df

# ============================================================================
# Basic Portfolio Metrics
# ============================================================================

def compute_metrics(df: pd.DataFrame, prices: Dict[str, float]) -> pd.DataFrame:
    """
    Compute basic portfolio metrics.
    
    Parameters
    ----------
    df : pd.DataFrame
        Portfolio data with Ticker, Purchase Price, Quantity columns
    prices : Dict[str, float]
        Current prices by ticker
        
    Returns
    -------
    pd.DataFrame
        DataFrame with computed basic metrics
    """
    if df is None or df.empty:
        return pd.DataFrame()
    
    try:
        result_df = df.copy()
        
        # Validate required columns
        required_cols = ['Ticker', 'Purchase Price', 'Quantity']
        missing_cols = [col for col in required_cols if col not in result_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Current prices
        result_df['Current Price'] = result_df['Ticker'].apply(
            lambda t: prices.get(str(t).upper(), np.nan)
        )
        
        # Basic calculations
        result_df['Total Value'] = result_df['Current Price'] * result_df['Quantity']
        result_df['Cost Basis'] = result_df['Purchase Price'] * result_df['Quantity']
        result_df['P/L'] = result_df['Total Value'] - result_df['Cost Basis']
        
        # P/L percentage
        result_df['P/L %'] = np.where(
            result_df['Cost Basis'] > 0,
            (result_df['Total Value'] / result_df['Cost Basis'] - 1.0) * 100.0,
            np.nan
        )
        
        # Portfolio weights
        total_value = result_df['Total Value'].sum()
        if total_value > 0:
            result_df['Weight %'] = result_df['Total Value'] / total_value * 100.0
        else:
            result_df['Weight %'] = np.nan
        
        logger.info(f"Computed metrics for {len(result_df)} assets")
        return result_df
        
    except Exception as e:
        logger.error(f"Error computing basic metrics: {e}")
        return df.copy() if df is not None else pd.DataFrame()

# ============================================================================
# Advanced Financial Metrics
# ============================================================================

def compute_rsi(prices: pd.Series, period: int = 14) -> float:
    """
    Compute Relative Strength Index (RSI).
    
    Parameters
    ----------
    prices : pd.Series
        Historical price series
    period : int, default 14
        RSI calculation period
        
    Returns
    -------
    float
        Current RSI value
    """
    if prices is None or len(prices) < period + 1:
        return np.nan
    
    try:
        # Calculate price changes
        delta = prices.diff().dropna()
        
        # Separate gains and losses
        gains = delta.clip(lower=0)
        losses = -delta.clip(upper=0)
        
        # Calculate average gains and losses using exponential moving average
        avg_gain = gains.ewm(alpha=1.0/period, min_periods=period).mean()
        avg_loss = losses.ewm(alpha=1.0/period, min_periods=period).mean()
        
        # Calculate RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi.iloc[-1]) if not rsi.empty else np.nan
        
    except Exception as e:
        logger.debug(f"Error computing RSI: {e}")
        return np.nan

def compute_volatility(prices: pd.Series, annualize: bool = True) -> float:
    """
    Compute price volatility (standard deviation of returns).
    
    Parameters
    ----------
    prices : pd.Series
        Historical price series
    annualize : bool, default True
        Whether to annualize the volatility
        
    Returns
    -------
    float
        Volatility as percentage
    """
    if prices is None or len(prices) < 2:
        return np.nan
    
    try:
        # Calculate daily returns
        returns = prices.pct_change().dropna()
        
        if returns.empty:
            return np.nan
        
        # Calculate volatility
        daily_vol = returns.std()
        
        if annualize:
            # Annualize using square root of trading days
            annual_vol = daily_vol * np.sqrt(252)
        else:
            annual_vol = daily_vol
        
        return float(annual_vol * 100.0)  # Convert to percentage
        
    except Exception as e:
        logger.debug(f"Error computing volatility: {e}")
        return np.nan

def compute_beta_alpha(
    asset_prices: pd.Series, 
    benchmark_prices: pd.Series,
    risk_free_rate: float = RISK_FREE_RATE
) -> Tuple[float, float]:
    """
    Compute Beta and Alpha relative to a benchmark.
    
    Parameters
    ----------
    asset_prices : pd.Series
        Asset price series
    benchmark_prices : pd.Series
        Benchmark price series
    risk_free_rate : float, default RISK_FREE_RATE
        Risk-free rate for Alpha calculation
        
    Returns
    -------
    Tuple[float, float]
        Beta and Alpha values
    """
    try:
        if asset_prices.empty or benchmark_prices.empty:
            return np.nan, np.nan
        
        # Calculate returns
        asset_returns = asset_prices.pct_change().dropna()
        benchmark_returns = benchmark_prices.pct_change().dropna()
        
        # Align the series
        aligned_data = pd.concat([asset_returns, benchmark_returns], axis=1, join='inner').dropna()
        
        if aligned_data.empty or len(aligned_data) < 2:
            return np.nan, np.nan
        
        asset_ret = aligned_data.iloc[:, 0]
        bench_ret = aligned_data.iloc[:, 1]
        
        # Calculate Beta
        covariance = np.cov(asset_ret, bench_ret)[0, 1]
        benchmark_variance = np.var(bench_ret)
        
        if benchmark_variance == 0:
            beta = np.nan
        else:
            beta = covariance / benchmark_variance
        
        # Calculate Alpha
        asset_mean_return = asset_ret.mean()
        benchmark_mean_return = bench_ret.mean()
        
        # Convert annual risk-free rate to daily
        daily_rf_rate = risk_free_rate / 252
        
        expected_return = daily_rf_rate + beta * (benchmark_mean_return - daily_rf_rate)
        alpha = asset_mean_return - expected_return
        
        return float(beta), float(alpha)
        
    except Exception as e:
        logger.debug(f"Error computing Beta/Alpha: {e}")
        return np.nan, np.nan

def compute_enhanced_metrics(
    df: pd.DataFrame,
    prices: Dict[str, float],
    benchmark_data: Optional[pd.DataFrame] = None,
    period: str = "6mo"
) -> pd.DataFrame:
    """
    Compute comprehensive portfolio metrics including technical indicators.
    
    Parameters
    ----------
    df : pd.DataFrame
        Portfolio data
    prices : Dict[str, float]
        Current prices
    benchmark_data : pd.DataFrame, optional
        Benchmark price data
    period : str, default "6mo"
        Historical period for calculations
        
    Returns
    -------
    pd.DataFrame
        DataFrame with enhanced metrics
    """
    # Start with basic metrics
    result_df = compute_metrics(df, prices)
    
    if result_df.empty:
        return result_df
    
    # Initialize advanced metric columns
    advanced_cols = ['RSI', 'Volatility', 'Beta', 'Alpha']
    for col in advanced_cols:
        result_df[col] = np.nan
    
    # Get benchmark returns for Beta/Alpha calculations
    benchmark_returns = None
    if benchmark_data is not None and not benchmark_data.empty:
        primary_benchmark = benchmark_data.iloc[:, 0]
        if not primary_benchmark.empty:
            benchmark_returns = primary_benchmark
    
    # Calculate advanced metrics for each asset
    for idx, row in result_df.iterrows():
        ticker = str(row.get('Ticker', '')).upper().strip()
        if not ticker:
            continue
        
        try:
            # Fetch historical prices
            hist_prices = fetch_historical_series(ticker, period=period)
            
            if not hist_prices.empty:
                # RSI
                rsi = compute_rsi(hist_prices)
                result_df.at[idx, 'RSI'] = rsi
                
                # Volatility
                volatility = compute_volatility(hist_prices)
                result_df.at[idx, 'Volatility'] = volatility
                
                # Beta and Alpha
                if benchmark_returns is not None:
                    beta, alpha = compute_beta_alpha(hist_prices, benchmark_returns)
                    result_df.at[idx, 'Beta'] = beta
                    result_df.at[idx, 'Alpha'] = alpha
                    
        except Exception as e:
            logger.debug(f"Error computing advanced metrics for {ticker}: {e}")
            continue
    
    logger.info(f"Computed enhanced metrics for {len(result_df)} assets")
    return result_df

# ============================================================================
# Portfolio-Level Risk Measures
# ============================================================================

def calculate_portfolio_sharpe(
    metrics_df: pd.DataFrame,
    period: str = "6mo",
    risk_free_rate: float = RISK_FREE_RATE
) -> float:
    """
    Calculate portfolio Sharpe ratio.
    
    Parameters
    ----------
    metrics_df : pd.DataFrame
        Portfolio metrics with weights
    period : str, default "6mo"
        Historical period
    risk_free_rate : float, default RISK_FREE_RATE
        Annual risk-free rate
        
    Returns
    -------
    float
        Annualized Sharpe ratio
    """
    try:
        if metrics_df.empty or 'Weight %' not in metrics_df.columns:
            return np.nan
        
        # Get valid assets with weights
        valid_assets = metrics_df.dropna(subset=['Weight %', 'Ticker'])
        if valid_assets.empty:
            return np.nan
        
        weights = valid_assets['Weight %'] / 100.0
        tickers = valid_assets['Ticker'].str.upper().tolist()
        
        # Fetch historical data for all assets
        price_data = {}
        for ticker in tickers:
            prices = fetch_historical_series(ticker, period=period)
            if not prices.empty:
                price_data[ticker] = prices
        
        if not price_data:
            return np.nan
        
        # Create aligned price DataFrame
        price_df = pd.DataFrame(price_data).dropna()
        if price_df.empty:
            return np.nan
        
        # Calculate returns
        returns_df = price_df.pct_change().dropna()
        
        # Align weights with available data
        available_tickers = returns_df.columns.tolist()
        weight_mask = valid_assets['Ticker'].isin(available_tickers)
        aligned_weights = weights[weight_mask]
        aligned_weights = aligned_weights / aligned_weights.sum()  # Renormalize
        
        # Calculate portfolio returns
        portfolio_returns = (returns_df[available_tickers] * aligned_weights.values).sum(axis=1)
        
        # Calculate Sharpe ratio
        daily_rf_rate = risk_free_rate / 252
        excess_returns = portfolio_returns - daily_rf_rate
        
        if excess_returns.std() == 0:
            return np.nan
        
        sharpe_daily = excess_returns.mean() / excess_returns.std()
        sharpe_annual = sharpe_daily * np.sqrt(252)
        
        return float(sharpe_annual)
        
    except Exception as e:
        logger.error(f"Error calculating portfolio Sharpe ratio: {e}")
        return np.nan

def calculate_value_at_risk(
    metrics_df: pd.DataFrame,
    confidence: float = 0.95,
    period: str = "6mo"
) -> float:
    """
    Calculate portfolio Value at Risk (VaR).
    
    Parameters
    ----------
    metrics_df : pd.DataFrame
        Portfolio metrics with weights and values
    confidence : float, default 0.95
        Confidence level
    period : str, default "6mo"
        Historical period
        
    Returns
    -------
    float
        VaR in dollar terms
    """
    try:
        if metrics_df.empty or 'Weight %' not in metrics_df.columns or 'Total Value' not in metrics_df.columns:
            return np.nan
        
        # Get valid assets
        valid_assets = metrics_df.dropna(subset=['Weight %', 'Ticker', 'Total Value'])
        if valid_assets.empty:
            return np.nan
        
        weights = valid_assets['Weight %'] / 100.0
        tickers = valid_assets['Ticker'].str.upper().tolist()
        total_portfolio_value = valid_assets['Total Value'].sum()
        
        # Fetch historical data
        price_data = {}
        for ticker in tickers:
            prices = fetch_historical_series(ticker, period=period)
            if not prices.empty:
                price_data[ticker] = prices
        
        if not price_data:
            return np.nan
        
        # Create aligned returns
        price_df = pd.DataFrame(price_data).dropna()
        returns_df = price_df.pct_change().dropna()
        
        if returns_df.empty:
            return np.nan
        
        # Align weights
        available_tickers = returns_df.columns.tolist()
        weight_mask = valid_assets['Ticker'].isin(available_tickers)
        aligned_weights = weights[weight_mask]
        aligned_weights = aligned_weights / aligned_weights.sum()
        
        # Calculate portfolio returns
        portfolio_returns = (returns_df[available_tickers] * aligned_weights.values).sum(axis=1)
        
        # Calculate VaR
        var_percentile = (1 - confidence) * 100
        var_return = np.percentile(portfolio_returns, var_percentile)
        var_dollar = abs(var_return * total_portfolio_value)
        
        return float(var_dollar)
        
    except Exception as e:
        logger.error(f"Error calculating VaR: {e}")
        return np.nan

# ============================================================================
# Portfolio Analysis Functions
# ============================================================================

def asset_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize portfolio by asset type.
    
    Parameters
    ----------
    df : pd.DataFrame
        Portfolio data with Asset Type and Total Value columns
        
    Returns
    -------
    pd.DataFrame
        Summary by asset type
    """
    if df is None or df.empty:
        return pd.DataFrame()
    
    required_cols = ['Asset Type', 'Total Value']
    if not all(col in df.columns for col in required_cols):
        return pd.DataFrame()
    
    try:
        summary = df.groupby('Asset Type')['Total Value'].sum().reset_index()
        summary = summary.sort_values('Total Value', ascending=False)
        return summary
        
    except Exception as e:
        logger.error(f"Error creating asset breakdown: {e}")
        return pd.DataFrame()

def top_and_worst_assets(df: pd.DataFrame, n: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identify top and worst performing assets.
    
    Parameters
    ----------
    df : pd.DataFrame
        Portfolio data with P/L % column
    n : int, default 3
        Number of assets to return
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Top performers and worst performers
    """
    if df is None or df.empty or 'P/L %' not in df.columns:
        return pd.DataFrame(), pd.DataFrame()
    
    try:
        # Filter out NaN values
        valid_df = df.dropna(subset=['P/L %'])
        
        if valid_df.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        sorted_df = valid_df.sort_values('P/L %', ascending=False)
        
        top_n = sorted_df.head(n)
        worst_n = sorted_df.tail(n).iloc[::-1]  # Reverse to show worst first
        
        return top_n, worst_n
        
    except Exception as e:
        logger.error(f"Error identifying top/worst assets: {e}")
        return pd.DataFrame(), pd.DataFrame()

# ============================================================================
# Recommendations and Suggestions
# ============================================================================

def suggest_diversification(df: pd.DataFrame) -> Optional[str]:
    """
    Provide diversification suggestions based on asset allocation.
    
    Parameters
    ----------
    df : pd.DataFrame
        Portfolio data with Weight % and Asset Type
        
    Returns
    -------
    Optional[str]
        Diversification suggestion or None
    """
    if df is None or df.empty:
        return None
    
    required_cols = ['Asset Type', 'Weight %']
    if not all(col in df.columns for col in required_cols):
        return None
    
    try:
        # Calculate allocation by asset type
        allocation = df.groupby('Asset Type')['Weight %'].sum()
        
        if allocation.empty:
            return None
        
        max_allocation = allocation.max()
        dominant_type = allocation.idxmax()
        
        if max_allocation > 70:
            return (
                f"Your portfolio is heavily concentrated in {dominant_type} ({max_allocation:.1f}%). "
                f"Consider diversifying into other asset classes to reduce risk."
            )
        elif max_allocation > 50:
            return (
                f"You have significant exposure to {dominant_type} ({max_allocation:.1f}%). "
                f"Monitor this allocation and consider rebalancing if it grows further."
            )
        
        return None
        
    except Exception as e:
        logger.error(f"Error suggesting diversification: {e}")
        return None

def generate_portfolio_recommendations(metrics_df: pd.DataFrame) -> List[Dict[str, str]]:
    """
    Generate comprehensive portfolio recommendations.
    
    Parameters
    ----------
    metrics_df : pd.DataFrame
        Portfolio with enhanced metrics
        
    Returns
    -------
    List[Dict[str, str]]
        List of recommendation objects
    """
    recommendations = []
    
    if metrics_df is None or metrics_df.empty:
        return recommendations
    
    try:
        # Diversification check
        diversification_msg = suggest_diversification(metrics_df)
        if diversification_msg:
            recommendations.append({
                "type": "warning",
                "title": "Diversification",
                "description": diversification_msg
            })
        
        # High volatility check
        if 'Volatility' in metrics_df.columns:
            high_vol_assets = metrics_df[metrics_df['Volatility'] > 40]
            if not high_vol_assets.empty:
                tickers = ", ".join(high_vol_assets['Ticker'].astype(str).tolist()[:5])
                recommendations.append({
                    "type": "warning",
                    "title": "High Volatility Alert",
                    "description": f"These assets show high volatility (>40%): {tickers}. Consider position sizing carefully."
                })
        
        # RSI-based recommendations
        if 'RSI' in metrics_df.columns:
            oversold = metrics_df[metrics_df['RSI'] < 30]
            overbought = metrics_df[metrics_df['RSI'] > 70]
            
            if not oversold.empty:
                tickers = ", ".join(oversold['Ticker'].astype(str).tolist()[:3])
                recommendations.append({
                    "type": "info",
                    "title": "Potentially Oversold Assets",
                    "description": f"These assets may be oversold (RSI < 30): {tickers}. Research for potential opportunities."
                })
            
            if not overbought.empty:
                tickers = ", ".join(overbought['Ticker'].astype(str).tolist()[:3])
                recommendations.append({
                    "type": "info",
                    "title": "Potentially Overbought Assets",
                    "description": f"These assets may be overbought (RSI > 70): {tickers}. Monitor for potential corrections."
                })
        
        # Large loss check
        if 'P/L %' in metrics_df.columns:
            large_losses = metrics_df[metrics_df['P/L %'] < -20]
            if not large_losses.empty:
                tickers = ", ".join(large_losses['Ticker'].astype(str).tolist()[:3])
                recommendations.append({
                    "type": "warning",
                    "title": "Significant Losses",
                    "description": f"These positions have losses >20%: {tickers}. Review your investment thesis."
                })
        
        # Beta analysis
        if 'Beta' in metrics_df.columns:
            high_beta = metrics_df[metrics_df['Beta'] > 1.5]
            if not high_beta.empty:
                recommendations.append({
                    "type": "info",
                    "title": "High Beta Exposure",
                    "description": f"You have {len(high_beta)} high-beta assets (>1.5). These may be more volatile than the market."
                })
        
        # If no specific recommendations, provide general positive feedback
        if not recommendations:
            recommendations.append({
                "type": "success",
                "title": "Portfolio Health",
                "description": "Your portfolio appears well-balanced with no major red flags detected."
            })
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        return [{
            "type": "info",
            "title": "Analysis Unavailable",
            "description": "Unable to generate recommendations at this time. Please try refreshing your data."
        }]

def suggest_rebalancing(metrics_df: pd.DataFrame) -> Optional[Dict[str, pd.DataFrame]]:
    """
    Suggest portfolio rebalancing based on target allocations.
    
    Parameters
    ----------
    metrics_df : pd.DataFrame
        Portfolio data with Asset Type and Weight %
        
    Returns
    -------
    Optional[Dict[str, pd.DataFrame]]
        Current and suggested allocations
    """
    if metrics_df is None or metrics_df.empty:
        return None
    
    required_cols = ['Asset Type', 'Weight %']
    if not all(col in metrics_df.columns for col in required_cols):
        return None
    
    try:
        # Current allocation
        current = metrics_df.groupby('Asset Type')['Weight %'].sum().reset_index()
        current.columns = ['asset_type', 'weight']
        
        # Define target weights (these are example targets)
        asset_types = current['asset_type'].tolist()
        target_weights = {}
        
        for asset_type in asset_types:
            asset_lower = asset_type.lower()
            if 'stock' in asset_lower or 'equity' in asset_lower:
                target_weights[asset_type] = 50.0
            elif 'etf' in asset_lower:
                target_weights[asset_type] = 20.0
            elif 'bond' in asset_lower or 'fixed' in asset_lower:
                target_weights[asset_type] = 15.0
            elif 'crypto' in asset_lower or 'bitcoin' in asset_lower:
                target_weights[asset_type] = 5.0
            elif 'reit' in asset_lower:
                target_weights[asset_type] = 5.0
            else:
                target_weights[asset_type] = 5.0
        
        # Normalize target weights
        total_target = sum(target_weights.values())
        for asset_type in target_weights:
            target_weights[asset_type] = (target_weights[asset_type] / total_target) * 100.0
        
        # Create suggested DataFrame
        suggested = pd.DataFrame({
            'asset_type': list(target_weights.keys()),
            'weight': list(target_weights.values())
        })
        
        return {
            'current': current,
            'suggested': suggested
        }
        
    except Exception as e:
        logger.error(f"Error suggesting rebalancing: {e}")
        return None

# ============================================================================
# Validation and Utility Functions
# ============================================================================

def validate_tickers(tickers: Iterable[str]) -> Dict[str, bool]:
    """
    Validate ticker symbols using yfinance.
    
    Parameters
    ----------
    tickers : Iterable[str]
        Ticker symbols to validate
        
    Returns
    -------
    Dict[str, bool]
        Mapping of ticker to validity
    """
    if not YF_AVAILABLE:
        logger.warning("yfinance not available for ticker validation")
        return {str(t).upper(): False for t in tickers}
    
    results = {}
    
    for ticker in tickers:
        ticker_clean = str(ticker).upper().strip()
        if not ticker_clean:
            results[ticker_clean] = False
            continue
        
        try:
            # Try to fetch minimal data to validate
            ticker_obj = yf.Ticker(ticker_clean)
            info = ticker_obj.info
            
            # Check if we got valid info
            if info and len(info) > 1:  # yfinance returns minimal dict for invalid tickers
                results[ticker_clean] = True
            else:
                results[ticker_clean] = False
                
        except Exception as e:
            logger.debug(f"Ticker validation failed for {ticker_clean}: {e}")
            results[ticker_clean] = False
    
    valid_count = sum(results.values())
    logger.info(f"Validated {valid_count}/{len(results)} tickers successfully")
    
    return results

def check_password_strength(password: str) -> str:
    """
    Assess password strength using multiple criteria.
    
    Parameters
    ----------
    password : str
        Password to evaluate
        
    Returns
    -------
    str
        Strength level: 'Weak', 'Medium', or 'Strong'
    """
    if not password:
        return "Weak"
    
    score = 0
    
    # Length check
    if len(password) >= 8:
        score += 1
    if len(password) >= 12:
        score += 1
    
    # Character variety checks
    if any(c.islower() for c in password):
        score += 1
    if any(c.isupper() for c in password):
        score += 1
    if any(c.isdigit() for c in password):
        score += 1
    if any(not c.isalnum() for c in password):
        score += 1
    
    # Determine strength
    if score <= 2:
        return "Weak"
    elif score <= 4:
        return "Medium"
    else:
        return "Strong"

def clear_all_caches():
    """Clear all cached data."""
    global PRICE_CACHE, CACHE_TIMESTAMPS, HIST_PRICES_CACHE, BENCHMARK_CACHE
    
    PRICE_CACHE.clear()
    CACHE_TIMESTAMPS.clear()
    HIST_PRICES_CACHE.clear()
    BENCHMARK_CACHE.clear()
    
    logger.info("All caches cleared")

# ============================================================================
# Batch Operations for Large Portfolios
# ============================================================================

def batch_fetch_prices(tickers: List[str], batch_size: int = MAX_BATCH_SIZE) -> Dict[str, float]:
    """
    Fetch prices in batches for large portfolios.
    
    Parameters
    ----------
    tickers : List[str]
        List of tickers to fetch
    batch_size : int, default MAX_BATCH_SIZE
        Maximum tickers per batch
        
    Returns
    -------
    Dict[str, float]
        Combined price results
    """
    all_prices = {}
    
    # Process in batches
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        logger.debug(f"Fetching batch {i//batch_size + 1}: {len(batch)} tickers")
        
        batch_prices = fetch_current_prices(batch)
        all_prices.update(batch_prices)
        
        # Small delay between batches to be respectful to API
        if i + batch_size < len(tickers):
            time.sleep(0.1)
    
    return all_prices

# ============================================================================
# Error Handling and Logging Utilities
# ============================================================================

def log_portfolio_stats(df: pd.DataFrame, action: str = "processed"):
    """Log portfolio statistics for debugging."""
    if df is None or df.empty:
        logger.info(f"Portfolio {action}: empty")
        return
    
    try:
        stats = {
            "assets": len(df),
            "asset_types": df['Asset Type'].nunique() if 'Asset Type' in df.columns else 0,
            "total_value": df['Total Value'].sum() if 'Total Value' in df.columns else 0
        }
        logger.info(f"Portfolio {action}: {stats}")
        
    except Exception as e:
        logger.error(f"Error logging portfolio stats: {e}")

# ============================================================================
# Module Initialization
# ============================================================================

def initialize_module():
    """Initialize the portfolio utils module."""
    try:
        _ensure_portfolio_dir()
        logger.info(f"Portfolio utils initialized - yfinance: {'available' if YF_AVAILABLE else 'unavailable'}")
        
    except Exception as e:
        logger.error(f"Error initializing portfolio utils: {e}")
        raise

# Initialize on import
initialize_module()
