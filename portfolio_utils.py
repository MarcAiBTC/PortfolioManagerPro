"""
Enhanced Portfolio Utilities Module - FIXED VERSION
==================================================

This module provides comprehensive portfolio management utilities including:
- Robust data fetching with intelligent caching
- Advanced financial metrics calculation
- Portfolio analysis and recommendations
- File I/O operations with error handling
- Data validation and cleaning

Key fixes:
- Improved yfinance connection handling
- Better error recovery for failed tickers
- Enhanced caching with proper cleanup
- More robust price fetching with fallbacks
- Better NaN handling and data validation

Author: Enhanced by AI Assistant
"""

import os
import json
import logging
import time
import warnings
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional, Iterable, Union

import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress pandas warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

# Try to import yfinance with better error handling
try:
    import yfinance as yf
    YF_AVAILABLE = True
    logger.info("yfinance successfully imported")
    
    # Test yfinance connection immediately
    try:
        test_ticker = yf.Ticker("AAPL")
        test_info = test_ticker.fast_info
        logger.info("yfinance connection test successful")
    except Exception as e:
        logger.warning(f"yfinance connection test failed: {e}")
        
except ImportError as e:
    yf = None
    YF_AVAILABLE = False
    logger.error(f"yfinance not available: {e}")

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
MAX_BATCH_SIZE: int = 20  # Reduced for better reliability

# Request timeout in seconds
REQUEST_TIMEOUT: int = 10

# Maximum retries for failed requests
MAX_RETRIES: int = 3

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

# Track failed tickers to avoid repeated failures
FAILED_TICKERS: Dict[str, float] = {}  # ticker -> timestamp
FAILED_TICKER_TIMEOUT = 300  # 5 minutes

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
# Utility Functions
# ============================================================================

def clean_ticker(ticker: str) -> str:
    """Clean and standardize ticker symbols."""
    if not ticker or not isinstance(ticker, str):
        return ""
    
    # Remove whitespace and convert to uppercase
    clean = str(ticker).strip().upper()
    
    # Remove common prefixes/suffixes that might cause issues
    clean = clean.replace(" ", "")
    
    return clean

def is_ticker_failed(ticker: str) -> bool:
    """Check if a ticker has recently failed and should be skipped."""
    if ticker not in FAILED_TICKERS:
        return False
    
    # Check if enough time has passed since failure
    time_since_failure = time.time() - FAILED_TICKERS[ticker]
    return time_since_failure < FAILED_TICKER_TIMEOUT

def mark_ticker_failed(ticker: str) -> None:
    """Mark a ticker as failed with current timestamp."""
    FAILED_TICKERS[ticker] = time.time()

def clear_failed_tickers() -> None:
    """Clear the failed tickers cache."""
    global FAILED_TICKERS
    FAILED_TICKERS.clear()
    logger.info("Cleared failed tickers cache")

# ============================================================================
# Enhanced Price Fetching Functions
# ============================================================================

def fetch_single_price(ticker: str, retries: int = MAX_RETRIES) -> float:
    """
    Fetch current price for a single ticker with multiple fallback methods.
    
    Parameters
    ----------
    ticker : str
        Asset symbol to fetch
    retries : int, default MAX_RETRIES
        Number of retry attempts
        
    Returns
    -------
    float
        Current price or NaN if failed
    """
    if not YF_AVAILABLE:
        logger.warning("yfinance not available")
        return np.nan
    
    clean_tick = clean_ticker(ticker)
    if not clean_tick:
        logger.warning(f"Invalid ticker: {ticker}")
        return np.nan
    
    # Check if ticker recently failed
    if is_ticker_failed(clean_tick):
        logger.debug(f"Skipping recently failed ticker: {clean_tick}")
        return np.nan
    
    for attempt in range(retries):
        try:
            logger.debug(f"Fetching price for {clean_tick} (attempt {attempt + 1})")
            
            # Method 1: Try fast_info first (fastest)
            try:
                ticker_obj = yf.Ticker(clean_tick)
                fast_info = ticker_obj.fast_info
                
                if hasattr(fast_info, 'last_price') and fast_info.last_price:
                    price = float(fast_info.last_price)
                    if not pd.isna(price) and price > 0:
                        logger.debug(f"Got price via fast_info for {clean_tick}: {price}")
                        return price
                        
            except Exception as e:
                logger.debug(f"fast_info failed for {clean_tick}: {e}")
            
            # Method 2: Try recent history
            try:
                hist_data = yf.download(
                    clean_tick, 
                    period="1d", 
                    interval="1m",
                    progress=False,
                    show_errors=False,
                    timeout=REQUEST_TIMEOUT
                )
                
                if not hist_data.empty:
                    if 'Adj Close' in hist_data.columns:
                        price_series = hist_data['Adj Close'].dropna()
                    elif 'Close' in hist_data.columns:
                        price_series = hist_data['Close'].dropna()
                    else:
                        price_series = pd.Series(dtype=float)
                    
                    if not price_series.empty:
                        price = float(price_series.iloc[-1])
                        if not pd.isna(price) and price > 0:
                            logger.debug(f"Got price via history for {clean_tick}: {price}")
                            return price
                            
            except Exception as e:
                logger.debug(f"History download failed for {clean_tick}: {e}")
            
            # Method 3: Try info dictionary
            try:
                ticker_obj = yf.Ticker(clean_tick)
                info = ticker_obj.info
                
                price_keys = ['regularMarketPrice', 'currentPrice', 'previousClose', 'open']
                for key in price_keys:
                    if key in info and info[key]:
                        price = float(info[key])
                        if not pd.isna(price) and price > 0:
                            logger.debug(f"Got price via info[{key}] for {clean_tick}: {price}")
                            return price
                            
            except Exception as e:
                logger.debug(f"Info method failed for {clean_tick}: {e}")
            
            # If all methods failed on this attempt, wait before retry
            if attempt < retries - 1:
                time.sleep(0.5)
                
        except Exception as e:
            logger.debug(f"Overall attempt {attempt + 1} failed for {clean_tick}: {e}")
            if attempt < retries - 1:
                time.sleep(1)
    
    # Mark ticker as failed after all attempts
    mark_ticker_failed(clean_tick)
    logger.warning(f"Failed to fetch price for {clean_tick} after {retries} attempts")
    return np.nan

def fetch_current_prices(tickers: Iterable[str]) -> Dict[str, float]:
    """
    Fetch current prices for multiple tickers with improved error handling.
    
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
    
    tickers_list = [clean_ticker(t) for t in tickers if clean_ticker(t)]
    prices = {t: np.nan for t in tickers_list}
    
    if not tickers_list:
        return prices
    
    logger.info(f"Fetching prices for {len(tickers_list)} tickers")
    
    # Try batch fetch first for efficiency
    if len(tickers_list) <= MAX_BATCH_SIZE:
        try:
            batch_prices = _fetch_batch_prices(tickers_list)
            prices.update(batch_prices)
            
            # Check which tickers still need individual fetching
            failed_tickers = [t for t, p in prices.items() if pd.isna(p)]
            
            if failed_tickers:
                logger.info(f"Batch fetch incomplete, fetching {len(failed_tickers)} tickers individually")
                for ticker in failed_tickers:
                    if not is_ticker_failed(ticker):
                        individual_price = fetch_single_price(ticker)
                        if not pd.isna(individual_price):
                            prices[ticker] = individual_price
            
        except Exception as e:
            logger.warning(f"Batch fetch failed, falling back to individual: {e}")
            # Fall back to individual fetching
            for ticker in tickers_list:
                if not is_ticker_failed(ticker):
                    prices[ticker] = fetch_single_price(ticker)
    else:
        # Too many tickers, fetch individually
        logger.info("Too many tickers for batch, fetching individually")
        for ticker in tickers_list:
            if not is_ticker_failed(ticker):
                prices[ticker] = fetch_single_price(ticker)
    
    successful_count = sum(1 for p in prices.values() if not pd.isna(p))
    logger.info(f"Successfully fetched {successful_count}/{len(tickers_list)} prices")
    
    return prices

def _fetch_batch_prices(tickers_list: List[str]) -> Dict[str, float]:
    """Helper function to fetch prices in batch."""
    prices = {t: np.nan for t in tickers_list}
    
    try:
        tickers_str = " ".join(tickers_list)
        logger.debug(f"Batch fetching: {tickers_str}")
        
        data = yf.download(
            tickers=tickers_str, 
            period="1d", 
            interval="1m", 
            progress=False,
            show_errors=False,
            timeout=REQUEST_TIMEOUT
        )
        
        if data.empty:
            logger.warning("Batch download returned empty data")
            return prices
        
        # Handle multi-index columns (multiple tickers)
        if isinstance(data.columns, pd.MultiIndex):
            for ticker in tickers_list:
                try:
                    # Try Adj Close first, then Close
                    price_series = None
                    for price_col in ['Adj Close', 'Close']:
                        if (price_col, ticker) in data.columns:
                            price_series = data[(price_col, ticker)].dropna()
                            break
                    
                    if price_series is not None and not price_series.empty:
                        price = float(price_series.iloc[-1])
                        if not pd.isna(price) and price > 0:
                            prices[ticker] = price
                            
                except Exception as e:
                    logger.debug(f"Error processing batch data for {ticker}: {e}")
                    continue
        else:
            # Single ticker case
            if len(tickers_list) == 1:
                ticker = tickers_list[0]
                try:
                    price_series = None
                    for price_col in ['Adj Close', 'Close']:
                        if price_col in data.columns:
                            price_series = data[price_col].dropna()
                            break
                    
                    if price_series is not None and not price_series.empty:
                        price = float(price_series.iloc[-1])
                        if not pd.isna(price) and price > 0:
                            prices[ticker] = price
                            
                except Exception as e:
                    logger.debug(f"Error processing single ticker batch data: {e}")
        
        return prices
        
    except Exception as e:
        logger.warning(f"Batch fetch error: {e}")
        return prices

def get_cached_prices(
    tickers: Iterable[str], 
    cache_duration_minutes: int = CACHE_DURATION_MINUTES
) -> Dict[str, float]:
    """
    Get current prices with intelligent caching and improved error handling.
    
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
    tickers_list = [clean_ticker(t) for t in tickers if clean_ticker(t)]
    
    if not tickers_list:
        logger.warning("No valid tickers provided")
        return {}
    
    if not YF_AVAILABLE:
        logger.warning("yfinance not available")
        return {ticker: np.nan for ticker in tickers_list}
    
    # Create cache key
    cache_key = ",".join(sorted(tickers_list))
    now = time.time()
    
    # Check if we have valid cached data
    if cache_key in PRICE_CACHE and cache_key in CACHE_TIMESTAMPS:
        cache_age = now - CACHE_TIMESTAMPS[cache_key]
        if cache_age < cache_duration_minutes * 60:
            cached_prices = PRICE_CACHE[cache_key]
            # Verify cached data is complete and valid
            valid_cached = True
            for ticker in tickers_list:
                if ticker not in cached_prices or pd.isna(cached_prices[ticker]):
                    valid_cached = False
                    break
            
            if valid_cached:
                logger.debug(f"Using cached prices for {len(tickers_list)} tickers")
                return {t: cached_prices[t] for t in tickers_list}
            else:
                logger.debug("Cached data incomplete, fetching fresh")
    
    # Fetch fresh data
    logger.info(f"Fetching fresh prices for {len(tickers_list)} tickers")
    
    try:
        prices = fetch_current_prices(tickers_list)
        
        # Only cache if we got some valid prices
        valid_prices = {k: v for k, v in prices.items() if not pd.isna(v)}
        if valid_prices:
            PRICE_CACHE[cache_key] = prices
            CACHE_TIMESTAMPS[cache_key] = now
            logger.debug(f"Cached {len(valid_prices)} valid prices")
        
        return prices
        
    except Exception as e:
        logger.error(f"Error in get_cached_prices: {e}")
        return {ticker: np.nan for ticker in tickers_list}

# ============================================================================
# Portfolio File Operations (keeping existing functionality)
# ============================================================================

def list_portfolios(username: str) -> List[str]:
    """List portfolio files for a given user, sorted by modification time."""
    _ensure_portfolio_dir()
    files = []
    
    try:
        for fname in os.listdir(PORTFOLIO_DIR):
            if not (fname.endswith(".csv") or fname.endswith(".json")):
                continue
            if fname.startswith(f"{username}_"):
                files.append(fname)
        
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
    """Save a portfolio to disk with optional history keeping."""
    _ensure_portfolio_dir()
    
    if fmt not in {"csv", "json"}:
        raise ValueError(f"Unsupported format: {fmt}")
    
    try:
        current_fname = f"{username}_current.{fmt}"
        current_path = os.path.join(PORTFOLIO_DIR, current_fname)
        
        if overwrite:
            if fmt == "csv":
                df.to_csv(current_path, index=False)
            else:
                df.to_json(current_path, orient="records", indent=2)
            logger.info(f"Saved current portfolio for {username}: {current_path}")
        
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
            logger.info(f"Saved portfolio snapshot: {snapshot_path}")
        
        return written_path
        
    except Exception as e:
        logger.error(f"Error saving portfolio for {username}: {e}")
        raise

def load_portfolio(username: str, filename: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Load a portfolio from disk."""
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
# Basic Portfolio Metrics with Enhanced Error Handling
# ============================================================================

def compute_metrics(df: pd.DataFrame, prices: Dict[str, float]) -> pd.DataFrame:
    """
    Compute basic portfolio metrics with enhanced error handling.
    
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
        logger.warning("Empty or None DataFrame provided to compute_metrics")
        return pd.DataFrame()
    
    try:
        result_df = df.copy()
        
        # Validate required columns
        required_cols = ['Ticker', 'Purchase Price', 'Quantity']
        missing_cols = [col for col in required_cols if col not in result_df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Clean tickers
        result_df['Ticker'] = result_df['Ticker'].apply(clean_ticker)
        
        # Current prices with better error handling
        def get_price_safe(ticker):
            """Safely get price for a ticker."""
            try:
                clean_tick = clean_ticker(str(ticker))
                return prices.get(clean_tick, np.nan)
            except Exception as e:
                logger.debug(f"Error getting price for {ticker}: {e}")
                return np.nan
        
        result_df['Current Price'] = result_df['Ticker'].apply(get_price_safe)
        
        # Convert numeric columns with error handling
        numeric_cols = ['Purchase Price', 'Quantity', 'Current Price']
        for col in numeric_cols:
            if col in result_df.columns:
                result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
        
        # Basic calculations with NaN handling
        result_df['Total Value'] = result_df['Current Price'] * result_df['Quantity']
        result_df['Cost Basis'] = result_df['Purchase Price'] * result_df['Quantity']
        result_df['P/L'] = result_df['Total Value'] - result_df['Cost Basis']
        
        # P/L percentage with division by zero protection
        result_df['P/L %'] = np.where(
            (result_df['Cost Basis'] > 0) & (~pd.isna(result_df['Cost Basis'])),
            (result_df['Total Value'] / result_df['Cost Basis'] - 1.0) * 100.0,
            np.nan
        )
        
        # Portfolio weights with total value protection
        total_value = result_df['Total Value'].sum()
        if total_value > 0 and not pd.isna(total_value):
            result_df['Weight %'] = result_df['Total Value'] / total_value * 100.0
        else:
            result_df['Weight %'] = np.nan
        
        # Log statistics
        valid_prices = result_df['Current Price'].notna().sum()
        total_assets = len(result_df)
        logger.info(f"Computed metrics: {valid_prices}/{total_assets} assets have valid prices")
        
        if valid_prices == 0:
            logger.warning("No valid prices found - all assets will show NaN values")
        
        return result_df
        
    except Exception as e:
        logger.error(f"Error computing basic metrics: {e}")
        return df.copy() if df is not None else pd.DataFrame()

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
        return {clean_ticker(str(t)): False for t in tickers}
    
    results = {}
    
    for ticker in tickers:
        ticker_clean = clean_ticker(str(ticker))
        if not ticker_clean:
            results[ticker_clean] = False
            continue
        
        try:
            # Try to fetch minimal data to validate
            ticker_obj = yf.Ticker(ticker_clean)
            
            # Try fast_info first
            try:
                info = ticker_obj.fast_info
                if hasattr(info, 'last_price') or hasattr(info, 'previousClose'):
                    results[ticker_clean] = True
                    continue
            except:
                pass
            
            # Fallback to info
            try:
                info = ticker_obj.info
                if info and len(info) > 5:  # Valid tickers usually have substantial info
                    results[ticker_clean] = True
                else:
                    results[ticker_clean] = False
            except:
                results[ticker_clean] = False
                
        except Exception as e:
            logger.debug(f"Ticker validation failed for {ticker_clean}: {e}")
            results[ticker_clean] = False
    
    valid_count = sum(results.values())
    logger.info(f"Validated {valid_count}/{len(results)} tickers successfully")
    
    return results

def clear_all_caches():
    """Clear all cached data."""
    global PRICE_CACHE, CACHE_TIMESTAMPS, HIST_PRICES_CACHE, BENCHMARK_CACHE, FAILED_TICKERS
    
    PRICE_CACHE.clear()
    CACHE_TIMESTAMPS.clear()
    HIST_PRICES_CACHE.clear()
    BENCHMARK_CACHE.clear()
    FAILED_TICKERS.clear()
    
    logger.info("All caches cleared")

def get_cache_stats() -> Dict[str, int]:
    """Get cache statistics for debugging."""
    return {
        "price_cache_entries": len(PRICE_CACHE),
        "hist_cache_entries": len(HIST_PRICES_CACHE),
        "benchmark_cache_entries": len(BENCHMARK_CACHE),
        "failed_tickers": len(FAILED_TICKERS)
    }

# ============================================================================
# Test Functions for Debugging
# ============================================================================

def test_yfinance_connection() -> Dict[str, any]:
    """Test yfinance connection and functionality."""
    results = {
        "yfinance_available": YF_AVAILABLE,
        "connection_test": False,
        "sample_prices": {},
        "errors": []
    }
    
    if not YF_AVAILABLE:
        results["errors"].append("yfinance not installed or not available")
        return results
    
    # Test basic connection
    try:
        test_tickers = ["AAPL", "MSFT", "GOOGL"]
        logger.info(f"Testing connection with tickers: {test_tickers}")
        
        prices = fetch_current_prices(test_tickers)
        results["sample_prices"] = prices
        
        valid_prices = sum(1 for p in prices.values() if not pd.isna(p))
        results["connection_test"] = valid_prices > 0
        
        if valid_prices == 0:
            results["errors"].append("No valid prices returned from test tickers")
        else:
            logger.info(f"Connection test successful: {valid_prices}/{len(test_tickers)} prices")
            
    except Exception as e:
        results["errors"].append(f"Connection test failed: {str(e)}")
        logger.error(f"yfinance connection test failed: {e}")
    
    return results

# ============================================================================
# Historical Data Functions (Simplified)
# ============================================================================

def fetch_historical_series(
    ticker: str, 
    period: str = "6mo", 
    interval: str = "1d"
) -> pd.Series:
    """Fetch historical price series for a single ticker with caching."""
    if not YF_AVAILABLE:
        return pd.Series(dtype=float)
    
    ticker_clean = clean_ticker(ticker)
    cache_key = (ticker_clean, period)
    
    # Check cache
    if cache_key in HIST_PRICES_CACHE:
        return HIST_PRICES_CACHE[cache_key]
    
    try:
        logger.debug(f"Fetching historical data for {ticker_clean} ({period})")
        data = yf.download(
            ticker_clean, 
            period=period, 
            interval=interval, 
            progress=False,
            show_errors=False,
            timeout=REQUEST_TIMEOUT
        )
        
        if data.empty:
            series = pd.Series(dtype=float)
        else:
            if 'Adj Close' in data.columns:
                series = data['Adj Close'].dropna()
            elif 'Close' in data.columns:
                series = data['Close'].dropna()
            else:
                series = pd.Series(dtype=float)
        
        # Cache the result
        HIST_PRICES_CACHE[cache_key] = series
        logger.debug(f"Cached {len(series)} historical prices for {ticker_clean}")
        
        return series
        
    except Exception as e:
        logger.error(f"Error fetching historical data for {ticker_clean}: {e}")
        empty_series = pd.Series(dtype=float)
        HIST_PRICES_CACHE[cache_key] = empty_series
        return empty_series

# ============================================================================
# Module Initialization
# ============================================================================

def initialize_module():
    """Initialize the portfolio utils module."""
    try:
        _ensure_portfolio_dir()
        
        # Test yfinance if available
        if YF_AVAILABLE:
            test_results = test_yfinance_connection()
            if test_results["connection_test"]:
                logger.info("Portfolio utils initialized successfully - yfinance connection OK")
            else:
                logger.warning(f"Portfolio utils initialized with warnings: {test_results['errors']}")
        else:
            logger.warning("Portfolio utils initialized - yfinance unavailable")
        
    except Exception as e:
        logger.error(f"Error initializing portfolio utils: {e}")
        raise

# Initialize on import
initialize_module()
