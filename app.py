"""
Enhanced Financial Portfolio Manager - Main Application
=====================================================

A comprehensive Streamlit application for managing investment portfolios with 
advanced visualizations, real-time metrics, and intelligent analysis.

Key improvements:
- Fixed HTML rendering issues with proper unsafe_allow_html flags
- Better error handling and logging
- Modular code structure with clear separation of concerns
- Enhanced caching and performance optimization
- Improved UI/UX with modern design patterns

Author: Enhanced by AI Assistant
"""

import os
import time
import traceback
import logging
import warnings
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import custom modules
from auth import authenticate_user, register_user
import portfolio_utils as putils

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

# ============================================================================
# Configuration and Setup
# ============================================================================

st.set_page_config(
    page_title="📊 Portfolio Manager Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS styling with improved responsiveness
def load_custom_css():
    """Load custom CSS styles for the application."""
    st.markdown("""
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        /* Global styling */
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            text-align: center;
            padding: 2rem 1rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            font-family: 'Inter', sans-serif;
        }
        
        .main-header h1 {
            margin: 0;
            font-size: 2.5rem;
            font-weight: 700;
            text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .main-header p {
            margin: 0.5rem 0 0 0;
            font-size: 1.1rem;
            opacity: 0.9;
        }
        
        .metric-card {
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
            border: 1px solid #e2e8f0;
            color: #1a202c;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.05);
            margin: 0.5rem 0;
            transition: all 0.2s ease;
            font-family: 'Inter', sans-serif;
        }
        
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(0,0,0,0.1);
        }
        
        .metric-card h3 {
            margin: 0 0 0.5rem 0;
            font-size: 0.9rem;
            font-weight: 600;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .metric-value {
            font-size: 1.8rem;
            font-weight: 700;
            margin: 0.5rem 0;
            color: #1e293b;
        }
        
        .metric-delta {
            font-size: 0.9rem;
            font-weight: 500;
        }
        
        .metric-help {
            font-size: 0.8rem;
            color: #64748b;
            margin-top: 0.5rem;
        }
        
        .welcome-banner {
            background: linear-gradient(135deg, #f0f9ff 0%, #dbeafe 100%);
            border: 2px solid #3b82f6;
            padding: 2rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 16px rgba(59,130,246,0.1);
            font-family: 'Inter', sans-serif;
        }
        
        .welcome-banner h2 {
            color: #1e40af;
            margin-bottom: 1rem;
        }
        
        .success-badge {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            display: inline-block;
            margin: 0.25rem;
            font-size: 0.9rem;
            font-weight: 500;
            box-shadow: 0 2px 8px rgba(16,185,129,0.3);
        }
        
        .warning-badge {
            background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            display: inline-block;
            margin: 0.25rem;
            font-size: 0.9rem;
            font-weight: 500;
            box-shadow: 0 2px 8px rgba(245,158,11,0.3);
        }
        
        .info-tooltip {
            background-color: #f8fafc;
            border-left: 4px solid #3b82f6;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }
        
        .performance-positive {
            color: #10b981;
            font-weight: 600;
        }
        
        .performance-negative {
            color: #ef4444;
            font-weight: 600;
        }
        
        .sidebar-section {
            background-color: #f8fafc;
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
            border-left: 4px solid #3b82f6;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }
        
        .chart-container {
            background: white;
            padding: 1rem;
            border-radius: 12px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.05);
            margin: 1rem 0;
            border: 1px solid #e2e8f0;
        }
        
        /* Improved button styling */
        .stButton > button {
            border-radius: 8px;
            border: none;
            font-weight: 500;
            transition: all 0.2s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        
        /* Enhanced selectbox styling */
        .stSelectbox > div > div {
            border-radius: 8px;
        }
        
        /* Improved dataframe styling */
        .dataframe {
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .main-header h1 {
                font-size: 2rem;
            }
            
            .metric-card {
                margin: 0.25rem 0;
            }
        }
    </style>
    """, unsafe_allow_html=True)

# Load CSS
load_custom_css()

# ============================================================================
# Session State Management
# ============================================================================

def initialize_session_state():
    """Initialize all session state variables with enhanced defaults."""
    defaults = {
        'authenticated': False,
        'username': '',
        'portfolio_df': None,
        'selected_portfolio_file': None,
        'price_cache': {},
        'price_cache_time': 0,
        'first_login': True,
        'portfolio_modified': False,
        'show_welcome': True,
        'last_refresh': None,
        'benchmark_data': None,
        'education_mode': True,
        'selected_timeframe': '6mo',
        'app_version': '2.1.0'
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

initialize_session_state()

# ============================================================================
# UI Helper Functions
# ============================================================================

def show_tooltip(text: str, tooltip: str) -> str:
    """Display text with a tooltip if education mode is enabled."""
    if st.session_state.education_mode:
        return f"{text} ℹ️"
    return text

def create_metric_card(title: str, value: str, delta: str = None, help_text: str = None) -> str:
    """Create a styled metric card with optional delta and help."""
    delta_class = "performance-positive" if delta and not str(delta).startswith("-") else "performance-negative"
    delta_html = f'<div class="metric-delta {delta_class}">{delta}</div>' if delta else ""
    help_html = f'<div class="metric-help">{help_text}</div>' if help_text else ""
    
    return f"""
    <div class="metric-card">
        <h3>{title}</h3>
        <div class="metric-value">{value}</div>
        {delta_html}
        {help_html}
    </div>
    """

def show_error_with_details(error_msg: str, details: str = None):
    """Show error message with optional details in education mode."""
    st.error(f"❌ {error_msg}")
    
    if st.session_state.education_mode and details:
        with st.expander("🔍 Error Details"):
            st.code(details)

def safe_load_portfolio(username: str, filename: Optional[str] = None) -> bool:
    """Safely load portfolio with comprehensive error handling."""
    try:
        with st.spinner("📂 Loading portfolio..."):
            df = putils.load_portfolio(username, filename)
            
        if df is not None and not df.empty:
            # Validate required columns
            required_cols = ['Ticker', 'Purchase Price', 'Quantity', 'Asset Type']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                show_error_with_details(
                    f"Portfolio missing required columns: {', '.join(missing_cols)}",
                    f"Required columns: {required_cols}"
                )
                return False
            
            st.session_state.portfolio_df = df
            st.session_state.selected_portfolio_file = filename
            st.session_state.portfolio_modified = False
            st.session_state.last_refresh = datetime.now()
            
            st.success(f"✅ Portfolio loaded successfully! ({len(df)} assets)")
            logger.info(f"Portfolio loaded for user {username}: {len(df)} assets")
            return True
        else:
            st.warning("⚠️ Portfolio file is empty or could not be loaded")
            return False
            
    except Exception as e:
        error_msg = f"Error loading portfolio: {str(e)}"
        show_error_with_details(error_msg, traceback.format_exc())
        logger.error(f"Portfolio load failed for {username}: {e}")
        return False

# ============================================================================
# Welcome and Onboarding
# ============================================================================

def show_welcome_message():
    """Enhanced welcome message with better HTML rendering."""
    if st.session_state.show_welcome and st.session_state.authenticated:
        welcome_html = f"""
        <div class="welcome-banner">
            <h2>🎉 Welcome to Portfolio Manager Pro, {st.session_state.username}!</h2>
            <p><strong>Your comprehensive investment dashboard is ready!</strong></p>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-top: 1.5rem;">
                <div>
                    <h4>📊 What you can do:</h4>
                    <ul style="text-align: left; margin: 0; padding-left: 1.5rem;">
                        <li>📈 <strong>Track performance</strong> with real-time data</li>
                        <li>📋 <strong>Add assets</strong> manually or upload CSV/JSON</li>
                        <li>🎯 <strong>Analyze risk</strong> with Alpha, Beta, RSI metrics</li>
                        <li>📊 <strong>Visualize allocation</strong> with interactive charts</li>
                    </ul>
                </div>
                <div>
                    <h4>🚀 Quick Start:</h4>
                    <ol style="text-align: left; margin: 0; padding-left: 1.5rem;">
                        <li>Add some assets or upload a portfolio</li>
                        <li>Explore the interactive dashboards</li>
                        <li>Use tooltips (ℹ️) to learn about metrics</li>
                        <li>Check diversification recommendations</li>
                    </ol>
                </div>
            </div>
            
            <div style="margin-top: 1.5rem; padding: 1rem; background-color: rgba(255,255,255,0.8); border-radius: 8px;">
                💡 <strong>Pro Tip:</strong> Enable Education Mode in the sidebar to see helpful explanations throughout the app!
            </div>
        </div>
        """
        
        st.markdown(welcome_html, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("🎯 Got it, let's start!", type="primary"):
                st.session_state.show_welcome = False
                st.rerun()
        with col2:
            if st.button("📚 Keep learning mode on"):
                st.session_state.education_mode = True
                st.session_state.show_welcome = False
                st.rerun()

# ============================================================================
# Portfolio Overview and Dashboard
# ============================================================================

def display_portfolio_overview():
    """Enhanced portfolio overview with comprehensive visualizations."""
    st.markdown(
        '<div class="main-header"><h1>📊 Portfolio Dashboard</h1><p>Real-time analysis of your investments</p></div>',
        unsafe_allow_html=True
    )
    
    username = st.session_state.username
    
    # Portfolio Selection Section
    with st.container():
        st.subheader("🗂️ Portfolio Selection")
        portfolios = putils.list_portfolios(username)
        
        if portfolios:
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                default_index = 0
                if st.session_state.selected_portfolio_file in portfolios:
                    try:
                        default_index = portfolios.index(st.session_state.selected_portfolio_file)
                    except ValueError:
                        pass
                
                selected_file = st.selectbox(
                    "Select a portfolio to analyze:",
                    portfolios,
                    index=default_index,
                    help="Choose from your saved portfolios"
                )
                
                if selected_file != st.session_state.selected_portfolio_file:
                    safe_load_portfolio(username, selected_file)
            
            with col2:
                if st.button("🔄 Refresh Data", help="Update prices and recalculate metrics"):
                    clear_cache()
                    if st.session_state.portfolio_df is not None:
                        st.rerun()
            
            with col3:
                if st.button("📊 Quick Stats", help="Show portfolio summary"):
                    show_portfolio_quick_stats()
        else:
            display_empty_portfolio_guide()

    df = st.session_state.portfolio_df
    if df is None or df.empty:
        return

    # Fetch and process data
    try:
        with st.spinner("📡 Fetching real-time market data..."):
            metrics_df = fetch_and_compute_metrics(df)
            
        if metrics_df is None or metrics_df.empty:
            st.error("❌ Unable to fetch market data. Please try again later.")
            return
            
    except Exception as e:
        show_error_with_details("Error processing portfolio data", str(e))
        return

    # Display components
    display_portfolio_summary(metrics_df)
    display_dashboard_tabs(metrics_df)

def clear_cache():
    """Clear all cached data."""
    st.session_state.price_cache = {}
    st.session_state.price_cache_time = 0
    st.session_state.benchmark_data = None
    # Clear portfolio utils cache
    putils.PRICE_CACHE.clear()
    putils.CACHE_TIMESTAMPS.clear()
    putils.HIST_PRICES_CACHE.clear()

def show_portfolio_quick_stats():
    """Show quick portfolio statistics."""
    if st.session_state.portfolio_df is not None:
        df = st.session_state.portfolio_df
        last_refresh = st.session_state.last_refresh
        refresh_text = last_refresh.strftime('%H:%M') if last_refresh else 'Unknown'
        
        st.info(f"""
        **Portfolio Quick Stats:**
        - 🎯 Assets: {len(df)}
        - 📊 Asset Types: {df['Asset Type'].nunique()}
        - 📅 Last Updated: {refresh_text}
        - 💾 File: {st.session_state.selected_portfolio_file or 'Current'}
        """)

def fetch_and_compute_metrics(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Fetch market data and compute enhanced metrics."""
    try:
        tickers = df['Ticker'].tolist()
        
        # Check if yfinance is available
        if not putils.YF_AVAILABLE:
            st.error("❌ Yahoo Finance data is not available. Please check your internet connection.")
            return None
        
        # Get current prices
        price_dict = putils.get_cached_prices(tickers)
        
        # Check for failed price fetches
        failed_tickers = [t for t, p in price_dict.items() if pd.isna(p)]
        if failed_tickers:
            st.warning(f"⚠️ Could not fetch prices for: {', '.join(failed_tickers[:5])}" + 
                      (f" and {len(failed_tickers)-5} more" if len(failed_tickers) > 5 else ""))
        
        # Get benchmark data
        benchmark_data = putils.fetch_benchmark_data()
        st.session_state.benchmark_data = benchmark_data
        
        # Compute enhanced metrics
        metrics_df = putils.compute_enhanced_metrics(
            df, price_dict, benchmark_data, st.session_state.selected_timeframe
        )
        
        return metrics_df
        
    except Exception as e:
        logger.error(f"Error in fetch_and_compute_metrics: {e}")
        raise

def display_empty_portfolio_guide():
    """Guide for users with empty portfolios."""
    guide_html = """
    <div style="text-align: center; padding: 3rem; background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); border-radius: 15px; margin: 2rem 0; border: 1px solid #e2e8f0;">
        <h2 style="color: #1e293b; margin-bottom: 1rem;">🚀 Let's Build Your Portfolio!</h2>
        <p style="font-size: 1.2rem; margin-bottom: 2rem; color: #64748b;">Start tracking your investments with our comprehensive tools</p>
        
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 2rem; margin: 2rem 0;">
            <div style="background: white; padding: 2rem; border-radius: 12px; box-shadow: 0 4px 16px rgba(0,0,0,0.1); border: 1px solid #e2e8f0;">
                <h3 style="color: #3b82f6; margin-bottom: 1rem;">➕ Add Assets Manually</h3>
                <p style="color: #64748b;">Start by adding individual stocks, ETFs, crypto, or other assets one by one</p>
            </div>
            <div style="background: white; padding: 2rem; border-radius: 12px; box-shadow: 0 4px 16px rgba(0,0,0,0.1); border: 1px solid #e2e8f0;">
                <h3 style="color: #10b981; margin-bottom: 1rem;">📤 Upload Portfolio</h3>
                <p style="color: #64748b;">Import your existing portfolio from CSV or JSON files</p>
            </div>
            <div style="background: white; padding: 2rem; border-radius: 12px; box-shadow: 0 4px 16px rgba(0,0,0,0.1); border: 1px solid #e2e8f0;">
                <h3 style="color: #f59e0b; margin-bottom: 1rem;">📚 Learn as You Go</h3>
                <p style="color: #64748b;">Use Education Mode to understand metrics and make better decisions</p>
            </div>
        </div>
    </div>
    """
    st.markdown(guide_html, unsafe_allow_html=True)

def display_portfolio_summary(metrics_df: pd.DataFrame):
    """Enhanced portfolio summary with visual metrics."""
    st.subheader("📈 Portfolio Summary")
    
    # Calculate key metrics
    total_value = metrics_df['Total Value'].sum()
    total_cost = metrics_df['Cost Basis'].sum()
    total_pl = total_value - total_cost
    total_pl_pct = (total_pl / total_cost * 100) if total_cost > 0 else 0
    
    # Create metric cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(create_metric_card(
            "💰 Total Value",
            f"${total_value:,.2f}",
            help_text="Current market value of all holdings"
        ), unsafe_allow_html=True)
    
    with col2:
        pl_symbol = "📈" if total_pl >= 0 else "📉"
        delta_text = f"{total_pl_pct:+.2f}%" if not pd.isna(total_pl_pct) else "N/A"
        st.markdown(create_metric_card(
            f"{pl_symbol} Total P/L",
            f"${total_pl:,.2f}",
            delta_text,
            help_text="Profit/Loss vs purchase price"
        ), unsafe_allow_html=True)
    
    with col3:
        if not metrics_df['P/L %'].isna().all():
            best_performer = metrics_df.loc[metrics_df['P/L %'].idxmax(), 'Ticker']
            best_pl = metrics_df['P/L %'].max()
        else:
            best_performer = "N/A"
            best_pl = 0
            
        st.markdown(create_metric_card(
            "🏆 Best Performer",
            str(best_performer),
            f"+{best_pl:.1f}%" if best_pl > 0 else "N/A",
            help_text="Asset with highest return percentage"
        ), unsafe_allow_html=True)
    
    with col4:
        diversification_score = len(metrics_df['Asset Type'].unique())
        st.markdown(create_metric_card(
            "🎯 Diversification",
            f"{diversification_score} types",
            f"{len(metrics_df)} assets",
            help_text="Number of different asset classes"
        ), unsafe_allow_html=True)

def display_dashboard_tabs(metrics_df: pd.DataFrame):
    """Display the main dashboard tabs with all analysis."""
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Performance Analysis", 
        "🥧 Asset Allocation", 
        "📊 Risk Analysis", 
        "📋 Holdings Detail",
        "🎯 Recommendations"
    ])
    
    with tab1:
        display_performance_analysis(metrics_df)
    
    with tab2:
        display_allocation_analysis(metrics_df)
    
    with tab3:
        display_risk_analysis(metrics_df)
    
    with tab4:
        display_holdings_detail(metrics_df)
    
    with tab5:
        display_recommendations(metrics_df)

# ============================================================================
# Analysis Display Functions
# ============================================================================

def display_performance_analysis(metrics_df: pd.DataFrame):
    """Enhanced performance analysis with multiple chart types."""
    st.subheader("📊 Performance Analysis")
    
    if metrics_df.empty:
        st.info("No data available for performance analysis.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # P/L Distribution Chart
        top_performers = metrics_df.nlargest(10, 'P/L')
        if not top_performers.empty:
            fig = px.bar(
                top_performers,
                x='Ticker',
                y='P/L',
                color='P/L',
                color_continuous_scale=['red', 'yellow', 'green'],
                title="🏆 Top 10 Performers by Profit/Loss ($)",
                labels={'P/L': 'Profit/Loss ($)'}
            )
            fig.update_layout(
                height=400,
                showlegend=False,
                xaxis_title="Asset",
                yaxis_title="Profit/Loss ($)"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Performance percentage chart
        if not metrics_df['P/L %'].isna().all():
            top_pct_performers = metrics_df.nlargest(10, 'P/L %')
            fig = px.bar(
                top_pct_performers,
                x='Ticker',
                y='P/L %',
                color='P/L %',
                color_continuous_scale=['red', 'yellow', 'green'],
                title="📈 Top 10 Performers by Return (%)",
                labels={'P/L %': 'Return (%)'}
            )
            fig.update_layout(
                height=400,
                showlegend=False,
                xaxis_title="Asset",
                yaxis_title="Return (%)"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Risk vs Return Analysis
    if 'Alpha' in metrics_df.columns and 'Beta' in metrics_df.columns:
        display_risk_return_analysis(metrics_df)

def display_risk_return_analysis(metrics_df: pd.DataFrame):
    """Display risk vs return scatter plot analysis."""
    st.subheader("🎯 Risk vs Return Analysis")
    
    # Filter out NaN values for better visualization
    clean_df = metrics_df.dropna(subset=['Alpha', 'Beta'])
    
    if clean_df.empty:
        st.info("Insufficient data for risk-return analysis.")
        return
    
    fig = px.scatter(
        clean_df,
        x='Beta',
        y='Alpha',
        size='Total Value',
        color='P/L %',
        hover_name='Ticker',
        hover_data=['P/L', 'RSI', 'Volatility'],
        title="📊 Risk-Return Profile (Alpha vs Beta)",
        labels={'Beta': 'Beta (Market Risk)', 'Alpha': 'Alpha (Excess Return)'},
        color_continuous_scale='RdYlGn'
    )
    
    # Add quadrant lines
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=1, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    if st.session_state.education_mode:
        with st.expander("📚 Understanding Risk-Return Analysis"):
            st.markdown("""
            **Alpha (Y-axis):** Measures excess return vs benchmark
            - **Positive:** Outperforming the market
            - **Negative:** Underperforming the market
            
            **Beta (X-axis):** Measures volatility vs market
            - **Beta > 1:** More volatile than market
            - **Beta < 1:** Less volatile than market
            - **Beta = 1:** Moves with market
            
            **Ideal Quadrant:** High Alpha, Low Beta (top-left)
            """)

def display_allocation_analysis(metrics_df: pd.DataFrame):
    """Asset allocation visualizations."""
    st.subheader("🥧 Asset Allocation Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Asset Type Distribution
        allocation_by_type = metrics_df.groupby('Asset Type')['Total Value'].sum().reset_index()
        
        if not allocation_by_type.empty:
            fig = px.pie(
                allocation_by_type,
                values='Total Value',
                names='Asset Type',
                title="📊 Allocation by Asset Type",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top Holdings by Value
        top_holdings = metrics_df.nlargest(8, 'Total Value')
        
        if not top_holdings.empty:
            fig = px.pie(
                top_holdings,
                values='Total Value',
                names='Ticker',
                title="💰 Top Holdings by Value",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Concentration Analysis
    display_concentration_analysis(metrics_df)

def display_concentration_analysis(metrics_df: pd.DataFrame):
    """Display portfolio concentration analysis."""
    st.subheader("🎯 Portfolio Concentration")
    
    # Calculate concentration metrics
    total_value = metrics_df['Total Value'].sum()
    if total_value > 0:
        top_5_concentration = metrics_df.nlargest(5, 'Total Value')['Total Value'].sum() / total_value * 100
        top_10_concentration = metrics_df.nlargest(10, 'Total Value')['Total Value'].sum() / total_value * 100
    else:
        top_5_concentration = 0
        top_10_concentration = 0
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Top 5 Holdings", f"{top_5_concentration:.1f}%", 
                 help="Percentage of portfolio in top 5 holdings")
    
    with col2:
        st.metric("Top 10 Holdings", f"{top_10_concentration:.1f}%",
                 help="Percentage of portfolio in top 10 holdings")
    
    with col3:
        if 'Weight %' in metrics_df.columns:
            weights = metrics_df['Weight %'].fillna(0) / 100
            herfindahl_index = (weights ** 2).sum()
            concentration_level = "High" if herfindahl_index > 0.25 else "Medium" if herfindahl_index > 0.15 else "Low"
        else:
            concentration_level = "Unknown"
        
        st.metric("Concentration Risk", concentration_level,
                 help="Based on Herfindahl-Hirschman Index")
    
    if st.session_state.education_mode:
        with st.expander("📚 Understanding Portfolio Concentration"):
            st.markdown("""
            **Portfolio Concentration** measures how your investments are distributed:
            
            - **Low Concentration:** Well-diversified, lower risk
            - **High Concentration:** Few large positions, higher risk
            
            **Healthy Guidelines:**
            - Top 5 holdings: < 50% of portfolio
            - No single holding: > 20% of portfolio
            - Multiple asset types represented
            """)

def display_risk_analysis(metrics_df: pd.DataFrame):
    """Comprehensive risk analysis dashboard."""
    st.subheader("⚠️ Risk Analysis Dashboard")
    
    # Risk Metrics Overview
    display_risk_metrics_overview(metrics_df)
    
    # Risk Distribution Charts
    col1, col2 = st.columns(2)
    
    with col1:
        display_volatility_distribution(metrics_df)
    
    with col2:
        display_beta_distribution(metrics_df)
    
    # Risk Heatmap
    display_risk_heatmap(metrics_df)

def display_risk_metrics_overview(metrics_df: pd.DataFrame):
    """Display overview of risk metrics."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_beta = metrics_df['Beta'].mean() if 'Beta' in metrics_df.columns else 0
        risk_level = "High" if avg_beta > 1.2 else "Medium" if avg_beta > 0.8 else "Low"
        st.metric("Portfolio Beta", f"{avg_beta:.2f}", risk_level)
    
    with col2:
        avg_volatility = metrics_df['Volatility'].mean() if 'Volatility' in metrics_df.columns else 0
        st.metric("Avg Volatility", f"{avg_volatility:.1f}%")
    
    with col3:
        sharpe_ratio = putils.calculate_portfolio_sharpe(metrics_df)
        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
    
    with col4:
        var_95 = putils.calculate_value_at_risk(metrics_df, confidence=0.95)
        st.metric("VaR (95%)", f"${var_95:,.0f}")

def display_volatility_distribution(metrics_df: pd.DataFrame):
    """Display volatility distribution histogram."""
    if 'Volatility' in metrics_df.columns:
        clean_vol = metrics_df['Volatility'].dropna()
        if not clean_vol.empty:
            fig = px.histogram(
                x=clean_vol,
                nbins=20,
                title="📊 Volatility Distribution",
                labels={'x': 'Volatility (%)', 'y': 'Number of Assets'},
                color_discrete_sequence=['#3b82f6']
            )
            fig.add_vline(x=clean_vol.mean(), line_dash="dash", 
                         line_color="red", annotation_text="Average")
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

def display_beta_distribution(metrics_df: pd.DataFrame):
    """Display beta distribution histogram."""
    if 'Beta' in metrics_df.columns:
        clean_beta = metrics_df['Beta'].dropna()
        if not clean_beta.empty:
            fig = px.histogram(
                x=clean_beta,
                nbins=20,
                title="📊 Beta Distribution",
                labels={'x': 'Beta (Market Risk)', 'y': 'Number of Assets'},
                color_discrete_sequence=['#f59e0b']
            )
            fig.add_vline(x=1, line_dash="dash", line_color="gray", 
                         annotation_text="Market Beta")
            fig.add_vline(x=clean_beta.mean(), line_dash="dash", 
                         line_color="red", annotation_text="Portfolio Avg")
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

def display_risk_heatmap(metrics_df: pd.DataFrame):
    """Display risk metrics heatmap."""
    st.subheader("🔥 Risk Heatmap")
    
    risk_metrics = ['P/L %', 'Beta', 'Volatility', 'RSI']
    available_metrics = [col for col in risk_metrics if col in metrics_df.columns]
    
    if available_metrics:
        risk_data = metrics_df[['Ticker'] + available_metrics].set_index('Ticker')
        
        # Remove rows with all NaN values
        risk_data = risk_data.dropna(how='all')
        
        if not risk_data.empty:
            fig = px.imshow(
                risk_data.T,
                aspect='auto',
                color_continuous_scale='RdYlGn_r',
                title="🎯 Risk Metrics Heatmap by Asset",
                labels={'color': 'Risk Level'}
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            if st.session_state.education_mode:
                with st.expander("📚 Reading the Risk Heatmap"):
                    st.markdown("""
                    **Color Coding:**
                    - 🟢 **Green:** Lower risk/better performance
                    - 🟡 **Yellow:** Medium risk
                    - 🔴 **Red:** Higher risk/worse performance
                    
                    **What to look for:**
                    - Assets with many red cells need attention
                    - Diversification across green/yellow is healthy
                    - Extreme values (very red/green) indicate outliers
                    """)

def display_holdings_detail(metrics_df: pd.DataFrame):
    """Detailed holdings table with enhanced formatting."""
    st.subheader("📋 Detailed Holdings")
    
    # Display options
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        search_term = st.text_input("🔍 Search assets:", placeholder="Enter ticker or asset type...")
    
    with col2:
        sort_columns = ['Total Value', 'P/L %', 'P/L', 'Ticker', 'Weight %']
        available_sort_columns = [col for col in sort_columns if col in metrics_df.columns]
        sort_by = st.selectbox("📊 Sort by:", available_sort_columns)
    
    with col3:
        ascending = st.checkbox("Ascending order", value=False)
    
    # Filter and sort data
    display_df = metrics_df.copy()
    
    # Apply search filter
    if search_term:
        mask = (
            display_df['Ticker'].str.contains(search_term, case=False, na=False) |
            display_df['Asset Type'].str.contains(search_term, case=False, na=False)
        )
        display_df = display_df[mask]
    
    # Apply sorting
    if sort_by in display_df.columns:
        display_df = display_df.sort_values(sort_by, ascending=ascending)
    
    # Format the dataframe for display
    formatted_df = format_dataframe_for_display(display_df)
    
    # Display the table
    st.dataframe(
        formatted_df,
        use_container_width=True,
        height=400
    )
    
    # Export options
    display_export_options(metrics_df)

def format_dataframe_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Format dataframe columns for better display."""
    display_df = df.copy()
    
    # Format currency columns
    currency_cols = ['Purchase Price', 'Current Price', 'Total Value', 'Cost Basis', 'P/L']
    for col in currency_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A")
    
    # Format percentage columns
    percentage_cols = ['P/L %', 'Weight %', 'Volatility']
    for col in percentage_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
    
    # Format decimal columns
    decimal_cols = ['Alpha', 'Beta', 'RSI']
    for col in decimal_cols:
        if col in display_df.columns:
            if col == 'Alpha':
                display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
            elif col == 'Beta':
                display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
            elif col == 'RSI':
                display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
    
    return display_df

def display_export_options(metrics_df: pd.DataFrame):
    """Display export options for portfolio data."""
    st.subheader("💾 Export Data")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv_data = metrics_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📄 Download CSV",
            data=csv_data,
            file_name=f"{st.session_state.username}_portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime='text/csv'
        )
    
    with col2:
        json_data = metrics_df.to_json(orient="records", indent=2).encode('utf-8')
        st.download_button(
            label="📋 Download JSON",
            data=json_data,
            file_name=f"{st.session_state.username}_portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime='application/json'
        )
    
    with col3:
        if st.button("📊 Generate Report", help="Create detailed portfolio report"):
            generate_portfolio_report(metrics_df)

def generate_portfolio_report(metrics_df: pd.DataFrame):
    """Generate a comprehensive portfolio report."""
    st.info("📊 Generating comprehensive portfolio report...")
    
    # Create report summary
    total_value = metrics_df['Total Value'].sum()
    total_cost = metrics_df['Cost Basis'].sum()
    total_pl = total_value - total_cost
    total_pl_pct = (total_pl / total_cost * 100) if total_cost > 0 else 0
    
    report_data = {
        "report_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "username": st.session_state.username,
        "total_assets": len(metrics_df),
        "total_value": total_value,
        "total_cost": total_cost,
        "total_pl": total_pl,
        "total_pl_pct": total_pl_pct,
        "asset_breakdown": metrics_df.groupby('Asset Type')['Total Value'].sum().to_dict(),
        "top_performers": metrics_df.nlargest(5, 'P/L %')[['Ticker', 'P/L %']].to_dict('records'),
        "recommendations": putils.generate_portfolio_recommendations(metrics_df)
    }
    
    # Display report summary
    with st.expander("📊 Portfolio Report Summary", expanded=True):
        st.json(report_data)
    
    st.success("✅ Report generated successfully!")

def display_recommendations(metrics_df: pd.DataFrame):
    """Intelligent portfolio recommendations."""
    st.subheader("🎯 Portfolio Recommendations")
    
    recommendations = putils.generate_portfolio_recommendations(metrics_df)
    
    if recommendations:
        for i, rec in enumerate(recommendations):
            rec_type = rec.get('type', 'info')
            icon = {"warning": "⚠️", "success": "✅", "info": "💡"}.get(rec_type, "📌")
            
            badge_class = "warning-badge" if rec_type == "warning" else "success-badge" if rec_type == "success" else "info-tooltip"
            
            st.markdown(f"""
            <div class="{badge_class}">
                {icon} <strong>{rec['title']}</strong><br>
                {rec['description']}
            </div>
            """, unsafe_allow_html=True)
    
    # Rebalancing suggestions
    display_rebalancing_suggestions(metrics_df)

def display_rebalancing_suggestions(metrics_df: pd.DataFrame):
    """Display portfolio rebalancing suggestions."""
    st.subheader("⚖️ Rebalancing Analysis")
    
    rebalancing_data = putils.suggest_rebalancing(metrics_df)
    
    if rebalancing_data:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Current Allocation:**")
            current_fig = px.pie(
                rebalancing_data['current'],
                values='weight',
                names='asset_type',
                title="Current Distribution"
            )
            current_fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(current_fig, use_container_width=True)
        
        with col2:
            st.write("**Suggested Allocation:**")
            suggested_fig = px.pie(
                rebalancing_data['suggested'],
                values='weight',
                names='asset_type',
                title="Suggested Distribution"
            )
            suggested_fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(suggested_fig, use_container_width=True)

# ============================================================================
# Asset Management Pages
# ============================================================================

def add_asset_page():
    """Enhanced asset addition with improved UX and validation."""
    st.markdown(
        '<div class="main-header"><h1>➕ Add New Asset</h1><p>Expand your portfolio with new investments</p></div>',
        unsafe_allow_html=True
    )
    
    username = st.session_state.username
    df = st.session_state.portfolio_df
    
    # Asset Addition Form
    with st.form("add_asset_form", clear_on_submit=True):
        st.subheader("📝 Asset Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            ticker = st.text_input(
                show_tooltip("🎯 Ticker Symbol", "Stock symbol (e.g., AAPL, TSLA, BTC-USD)"),
                max_chars=12,
                help="Enter the trading symbol for your asset",
                placeholder="e.g., AAPL, MSFT, BTC-USD"
            ).strip().upper()
            
            purchase_price = st.number_input(
                show_tooltip("💰 Purchase Price ($)", "Price per share/unit when you bought it"),
                min_value=0.0,
                format="%.4f",
                step=0.01,
                help="Enter the price you paid per unit"
            )
            
            asset_type = st.selectbox(
                show_tooltip("📊 Asset Type", "Category helps with portfolio analysis"),
                ["Stock", "ETF", "Crypto", "Bond", "REIT", "Commodity", "Option", "Other"],
                help="Choose the category that best describes this asset"
            )
        
        with col2:
            quantity = st.number_input(
                show_tooltip("📦 Quantity", "Number of shares/units you own"),
                min_value=0.0,
                format="%.6f",
                step=0.001,
                help="Enter the number of units you purchased"
            )
            
            purchase_date = st.date_input(
                show_tooltip("📅 Purchase Date", "When you bought this asset"),
                value=datetime.now().date(),
                help="This helps calculate holding period returns"
            )
            
            notes = st.text_area(
                show_tooltip("📝 Notes (Optional)", "Any additional information"),
                placeholder="e.g., Part of tech diversification strategy...",
                help="Optional notes about this investment"
            )
        
        # Real-time validation and preview
        display_asset_preview(ticker, purchase_price, quantity)
        
        # Form submission
        submitted = st.form_submit_button("➕ Add Asset", type="primary", use_container_width=True)
        
        if submitted:
            handle_asset_submission(ticker, purchase_price, quantity, asset_type, purchase_date, notes, username, df)

def display_asset_preview(ticker: str, purchase_price: float, quantity: float):
    """Display real-time asset preview."""
    if ticker and purchase_price > 0 and quantity > 0:
        st.subheader("👀 Preview")
        
        cost_basis = purchase_price * quantity
        
        # Try to fetch current price for preview
        try:
            with st.spinner("🔍 Fetching current price..."):
                current_prices = putils.fetch_current_prices([ticker])
                current_price = current_prices.get(ticker)
            
            if pd.notna(current_price):
                current_value = current_price * quantity
                pl = current_value - cost_basis
                pl_pct = (pl / cost_basis * 100) if cost_basis > 0 else 0
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("💰 Cost Basis", f"${cost_basis:,.2f}")
                with col2:
                    st.metric("📊 Current Value", f"${current_value:,.2f}")
                with col3:
                    st.metric("📈 P/L", f"${pl:,.2f}", f"{pl_pct:+.2f}%")
                with col4:
                    st.metric("💲 Current Price", f"${current_price:.2f}")
            else:
                st.info(f"💡 Cost basis will be ${cost_basis:,.2f}. Current price will be fetched after adding.")
        
        except Exception:
            st.info(f"💡 Cost basis will be ${cost_basis:,.2f}. Price data will be fetched after adding.")

def handle_asset_submission(ticker: str, purchase_price: float, quantity: float, 
                          asset_type: str, purchase_date, notes: str, 
                          username: str, df: Optional[pd.DataFrame]):
    """Handle asset form submission with validation."""
    # Validation
    errors = []
    
    if not ticker:
        errors.append("Ticker symbol is required")
    elif len(ticker) < 1:
        errors.append("Ticker symbol too short")
    
    if quantity <= 0:
        errors.append("Quantity must be greater than zero")
    
    if purchase_price <= 0:
        errors.append("Purchase price must be greater than zero")
    
    if errors:
        for error in errors:
            st.error(f"❌ {error}")
        return
    
    # Add the asset
    try:
        new_asset = {
            'Ticker': ticker,
            'Purchase Price': purchase_price,
            'Quantity': quantity,
            'Asset Type': asset_type,
            'Purchase Date': purchase_date.strftime('%Y-%m-%d'),
            'Notes': notes
        }
        
        # Add to portfolio
        if df is None or df.empty:
            new_df = pd.DataFrame([new_asset])
        else:
            new_df = pd.concat([df, pd.DataFrame([new_asset])], ignore_index=True)
        
        # Save portfolio
        putils.save_portfolio(username, new_df, overwrite=True)
        
        # Update session state
        st.session_state.portfolio_df = new_df
        st.session_state.portfolio_modified = True
        
        st.success(f"🎉 Successfully added {ticker} to your portfolio!")
        logger.info(f"Asset added: {ticker} for user {username}")
        
    except Exception as e:
        show_error_with_details(f"Error adding asset: {str(e)}", traceback.format_exc())

def upload_portfolio_page():
    """Enhanced portfolio upload with better validation and preview."""
    st.markdown(
        '<div class="main-header"><h1>📤 Upload Portfolio</h1><p>Import your existing investment data</p></div>',
        unsafe_allow_html=True
    )
    
    username = st.session_state.username
    
    # File format guide
    display_file_format_guide()
    
    # File upload section
    handle_file_upload(username)

def display_file_format_guide():
    """Display file format requirements and templates."""
    with st.expander("📋 Supported File Formats & Requirements", expanded=True):
        st.markdown("""
        ### Required Columns:
        
        | Column | Description | Example |
        |--------|-------------|---------|
        | **Ticker** | Asset symbol | AAPL, TSLA, BTC-USD |
        | **Purchase Price** | Price per unit when bought | 150.00 |
        | **Quantity** | Number of units owned | 10 |
        | **Asset Type** | Category of investment | Stock, ETF, Crypto |
        
        ### Optional Columns:
        - **Purchase Date**: When you bought the asset
        - **Notes**: Additional information
        
        ### Supported Formats:
        - 📄 **CSV**: Comma-separated values
        - 📋 **JSON**: JavaScript Object Notation
        """)
        
        # Sample data for download
        sample_data = pd.DataFrame({
            'Ticker': ['AAPL', 'MSFT', 'TSLA', 'BTC-USD'],
            'Purchase Price': [150.00, 300.00, 800.00, 45000.00],
            'Quantity': [10, 5, 2, 0.1],
            'Asset Type': ['Stock', 'Stock', 'Stock', 'Crypto'],
            'Purchase Date': ['2023-01-15', '2023-02-20', '2023-03-10', '2023-04-01'],
            'Notes': ['Tech diversification', 'Blue chip holding', 'Growth play', 'Crypto exposure']
        })
        
        col1, col2 = st.columns(2)
        with col1:
            csv_sample = sample_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📄 Download CSV Template",
                csv_sample,
                "portfolio_template.csv",
                "text/csv"
            )
        
        with col2:
            json_sample = sample_data.to_json(orient="records", indent=2).encode('utf-8')
            st.download_button(
                "📋 Download JSON Template",
                json_sample,
                "portfolio_template.json",
                "application/json"
            )

def handle_file_upload(username: str):
    """Handle the file upload process."""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "📁 Select your portfolio file",
            type=["csv", "json"],
            help="Upload a CSV or JSON file containing your portfolio data"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacing
        merge_option = st.radio(
            "📥 Import Options:",
            ["🔄 Replace current portfolio", "➕ Add to current portfolio"],
            help="Choose whether to replace or merge with existing data"
        )
    
    if uploaded_file is not None:
        process_uploaded_file(uploaded_file, merge_option, username)

def process_uploaded_file(uploaded_file, merge_option: str, username: str):
    """Process the uploaded portfolio file."""
    try:
        # Parse the file
        with st.spinner("📖 Reading file..."):
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:  # JSON
                df = pd.read_json(uploaded_file)
        
        st.success(f"✅ File '{uploaded_file.name}' loaded successfully!")
        
        # Validate and clean data
        validated_df = validate_and_clean_portfolio_data(df)
        
        if validated_df is None or validated_df.empty:
            st.error("❌ No valid data remaining after validation")
            return
        
        # Show preview and import options
        display_upload_preview(validated_df, merge_option, username)
        
    except Exception as e:
        show_error_with_details(f"Error processing file: {str(e)}", traceback.format_exc())

def validate_and_clean_portfolio_data(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Validate and clean uploaded portfolio data."""
    # Check required columns
    required_cols = {'Ticker', 'Purchase Price', 'Quantity', 'Asset Type'}
    missing_cols = required_cols - set(df.columns)
    
    if missing_cols:
        st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
        return None
    
    # Data cleaning
    original_rows = len(df)
    
    # Clean and validate data
    df['Ticker'] = df['Ticker'].astype(str).str.strip().str.upper()
    df['Purchase Price'] = pd.to_numeric(df['Purchase Price'], errors='coerce')
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
    df['Asset Type'] = df['Asset Type'].astype(str).str.strip()
    
    # Remove invalid rows
    df = df.dropna(subset=['Ticker', 'Purchase Price', 'Quantity'])
    df = df[df['Purchase Price'] > 0]
    df = df[df['Quantity'] > 0]
    df = df[df['Ticker'].str.len() > 0]
    
    cleaned_rows = len(df)
    removed_rows = original_rows - cleaned_rows
    
    if removed_rows > 0:
        st.warning(f"⚠️ Removed {removed_rows} invalid rows during cleaning")
    
    return df

def display_upload_preview(df: pd.DataFrame, merge_option: str, username: str):
    """Display preview of uploaded data and import options."""
    st.subheader("👀 Data Preview")
    
    # Summary stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total Assets", len(df))
    
    with col2:
        total_cost = (df['Purchase Price'] * df['Quantity']).sum()
        st.metric("💰 Total Cost", f"${total_cost:,.2f}")
    
    with col3:
        unique_types = df['Asset Type'].nunique()
        st.metric("🎯 Asset Types", unique_types)
    
    with col4:
        avg_position_size = total_cost / len(df) if len(df) > 0 else 0
        st.metric("📈 Avg Position", f"${avg_position_size:,.2f}")
    
    # Data table preview
    st.dataframe(df, use_container_width=True, height=300)
    
    # Asset type breakdown
    if len(df) > 0:
        display_asset_breakdown_chart(df)
    
    # Import confirmation
    display_import_confirmation(df, merge_option, username)

def display_asset_breakdown_chart(df: pd.DataFrame):
    """Display asset type breakdown chart."""
    st.subheader("📊 Asset Type Breakdown")
    type_breakdown = df.groupby('Asset Type').agg({
        'Ticker': 'count',
        'Purchase Price': lambda x: (x * df.loc[x.index, 'Quantity']).sum()
    }).rename(columns={'Ticker': 'Count', 'Purchase Price': 'Total Value'})
    
    fig = px.bar(
        type_breakdown.reset_index(),
        x='Asset Type',
        y='Count',
        title="Assets by Type",
        color='Asset Type'
    )
    st.plotly_chart(fig, use_container_width=True)

def display_import_confirmation(df: pd.DataFrame, merge_option: str, username: str):
    """Display import confirmation options."""
    st.subheader("💾 Confirm Import")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if st.button("🚀 Import Portfolio", type="primary"):
            import_portfolio_data(df, merge_option, username)
    
    with col2:
        if st.button("🔍 Validate Tickers", help="Check if all tickers are valid"):
            validate_portfolio_tickers(df)

def import_portfolio_data(df: pd.DataFrame, merge_option: str, username: str):
    """Import the portfolio data."""
    try:
        overwrite = merge_option.startswith("🔄")
        
        if not overwrite and st.session_state.portfolio_df is not None:
            # Merge with existing portfolio
            existing_df = st.session_state.portfolio_df
            df = pd.concat([existing_df, df], ignore_index=True)
        
        # Save the portfolio
        putils.save_portfolio(username, df, overwrite=True)
        
        # Update session state
        st.session_state.portfolio_df = df
        st.session_state.portfolio_modified = True
        
        action = "replaced" if overwrite else "merged with existing portfolio"
        st.success(f"🎉 Portfolio {action} successfully! ({len(df)} total assets)")
        st.balloons()
        logger.info(f"Portfolio imported for user {username}: {len(df)} assets")
        
    except Exception as e:
        show_error_with_details(f"Error saving portfolio: {str(e)}", traceback.format_exc())

def validate_portfolio_tickers(df: pd.DataFrame):
    """Validate ticker symbols in the portfolio."""
    with st.spinner("🔍 Validating ticker symbols..."):
        tickers = df['Ticker'].unique().tolist()
        validation_results = putils.validate_tickers(tickers)
        
        valid_tickers = [t for t, valid in validation_results.items() if valid]
        invalid_tickers = [t for t, valid in validation_results.items() if not valid]
        
        if invalid_tickers:
            st.warning(f"⚠️ Invalid tickers found: {', '.join(invalid_tickers)}")
            if st.session_state.education_mode:
                with st.expander("🔧 Ticker Validation Help"):
                    st.markdown("""
                    **Why might tickers be invalid?**
                    - Ticker symbol might be incorrect or delisted
                    - Different exchanges use different formats
                    - Crypto tickers often need suffixes (e.g., BTC-USD)
                    - Some international stocks need exchange codes
                    """)
        else:
            st.success("✅ All tickers validated successfully!")

def history_page():
    """Enhanced portfolio history management."""
    st.markdown(
        '<div class="main-header"><h1>📚 Portfolio History</h1><p>Manage your saved portfolios</p></div>',
        unsafe_allow_html=True
    )
    
    username = st.session_state.username
    files = putils.list_portfolios(username)
    
    if not files:
        display_empty_history_message()
        return
    
    st.write(f"📊 You have **{len(files)}** saved portfolios:")
    
    # Portfolio management interface
    display_portfolio_management_interface(files, username)
    
    # Portfolio timeline if multiple files exist
    if len(files) > 1:
        display_portfolio_timeline(files)

def display_empty_history_message():
    """Display message when no portfolio history exists."""
    empty_history_html = """
    <div style="text-align: center; padding: 3rem; background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); border-radius: 15px; margin: 2rem 0; border: 1px solid #e2e8f0;">
        <h2 style="color: #1e293b;">📝 No Portfolio History Yet</h2>
        <p style="font-size: 1.2rem; margin-bottom: 2rem; color: #64748b;">Start building your investment tracking history!</p>
        
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1.5rem; margin: 2rem 0;">
            <div style="background: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 16px rgba(0,0,0,0.1); border: 1px solid #e2e8f0;">
                <h4 style="color: #10b981;">➕ Add Assets</h4>
                <p style="color: #64748b;">Start by adding individual investments</p>
            </div>
            <div style="background: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 16px rgba(0,0,0,0.1); border: 1px solid #e2e8f0;">
                <h4 style="color: #3b82f6;">📤 Upload Files</h4>
                <p style="color: #64748b;">Import from CSV or JSON files</p>
            </div>
            <div style="background: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 16px rgba(0,0,0,0.1); border: 1px solid #e2e8f0;">
                <h4 style="color: #f59e0b;">💾 Auto-Save</h4>
                <p style="color: #64748b;">Your portfolios are saved automatically</p>
            </div>
        </div>
    </div>
    """
    st.markdown(empty_history_html, unsafe_allow_html=True)

def display_portfolio_management_interface(files: List[str], username: str):
    """Display portfolio management interface."""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_file = st.selectbox(
            "🗂️ Select Portfolio:",
            files,
            format_func=lambda x: f"{'📍 ' if x == st.session_state.selected_portfolio_file else '📁 '}{x}",
            help="Choose a portfolio to manage"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacing
        col_a, col_b = st.columns(2)
        
        with col_a:
            if st.button("📂 Load", help="Load selected portfolio"):
                if safe_load_portfolio(username, selected_file):
                    st.success(f"✅ Loaded '{selected_file}'")
        
        with col_b:
            if st.button("🗑️ Delete", help="Delete selected portfolio"):
                handle_portfolio_deletion(selected_file, username)
    
    if selected_file:
        display_portfolio_details(selected_file)

def handle_portfolio_deletion(selected_file: str, username: str):
    """Handle portfolio deletion with confirmation."""
    if selected_file == st.session_state.selected_portfolio_file:
        st.error("❌ Cannot delete currently active portfolio")
    else:
        # Use session state to track deletion confirmation
        if f"confirm_delete_{selected_file}" not in st.session_state:
            st.session_state[f"confirm_delete_{selected_file}"] = False
        
        if not st.session_state[f"confirm_delete_{selected_file}"]:
            if st.button("⚠️ Confirm Delete", type="secondary", key=f"delete_confirm_{selected_file}"):
                st.session_state[f"confirm_delete_{selected_file}"] = True
                st.rerun()
        else:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Yes, Delete", type="primary", key=f"delete_yes_{selected_file}"):
                    try:
                        file_path = os.path.join(putils.PORTFOLIO_DIR, selected_file)
                        os.remove(file_path)
                        st.success(f"✅ Deleted '{selected_file}'")
                        # Clean up session state
                        del st.session_state[f"confirm_delete_{selected_file}"]
                        st.rerun()
                    except Exception as e:
                        show_error_with_details(f"Error deleting file: {e}")
            
            with col2:
                if st.button("❌ Cancel", key=f"delete_no_{selected_file}"):
                    del st.session_state[f"confirm_delete_{selected_file}"]
                    st.rerun()

def display_portfolio_details(selected_file: str):
    """Display detailed information about a portfolio file."""
    st.subheader(f"📄 Portfolio Details: {selected_file}")
    
    try:
        file_path = os.path.join(putils.PORTFOLIO_DIR, selected_file)
        
        if os.path.exists(file_path):
            # File metadata
            file_stats = os.stat(file_path)
            file_size = file_stats.st_size
            file_modified = datetime.fromtimestamp(file_stats.st_mtime)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📊 File Size", f"{file_size:,} bytes")
            
            with col2:
                st.metric("📅 Modified", file_modified.strftime("%Y-%m-%d"))
            
            with col3:
                st.metric("🕐 Time", file_modified.strftime("%H:%M:%S"))
            
            with col4:
                is_current = selected_file == st.session_state.selected_portfolio_file
                status = "📍 Active" if is_current else "📁 Stored"
                st.metric("📌 Status", status)
            
            # Portfolio preview
            display_portfolio_file_preview(file_path, selected_file)
    
    except Exception as e:
        show_error_with_details(f"Error loading file details: {e}")

def display_portfolio_file_preview(file_path: str, selected_file: str):
    """Display preview of portfolio file contents."""
    with st.expander("👀 Portfolio Preview", expanded=True):
        try:
            # Load portfolio data
            if selected_file.endswith('.csv'):
                preview_df = pd.read_csv(file_path)
            else:
                preview_df = pd.read_json(file_path)
            
            if not preview_df.empty:
                # Quick stats
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("🎯 Assets", len(preview_df))
                
                with col2:
                    if 'Purchase Price' in preview_df.columns and 'Quantity' in preview_df.columns:
                        total_cost = (preview_df['Purchase Price'] * preview_df['Quantity']).sum()
                        st.metric("💰 Total Cost", f"${total_cost:,.2f}")
                
                with col3:
                    if 'Asset Type' in preview_df.columns:
                        asset_types = preview_df['Asset Type'].nunique()
                        st.metric("📊 Asset Types", asset_types)
                
                # Data preview
                st.dataframe(preview_df, use_container_width=True, height=200)
                
                # Download options
                display_file_download_options(preview_df, selected_file)
            else:
                st.warning("⚠️ Portfolio file appears to be empty")
        
        except Exception as e:
            show_error_with_details(f"Error loading preview: {e}")

def display_file_download_options(preview_df: pd.DataFrame, selected_file: str):
    """Display download options for portfolio files."""
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV download
        csv_data = preview_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📄 Download as CSV",
            csv_data,
            f"{selected_file.replace('.json', '.csv')}",
            "text/csv"
        )
    
    with col2:
        # JSON download
        json_data = preview_df.to_json(orient="records", indent=2).encode('utf-8')
        st.download_button(
            "📋 Download as JSON",
            json_data,
            f"{selected_file.replace('.csv', '.json')}",
            "application/json"
        )

def display_portfolio_timeline(files: List[str]):
    """Display portfolio timeline visualization."""
    st.subheader("📈 Portfolio Timeline")
    
    timeline_data = []
    for file in files:
        try:
            file_path = os.path.join(putils.PORTFOLIO_DIR, file)
            file_stats = os.stat(file_path)
            
            # Try to load and calculate value
            try:
                df = pd.read_csv(file_path) if file.endswith('.csv') else pd.read_json(file_path)
                if 'Purchase Price' in df.columns and 'Quantity' in df.columns:
                    total_value = (df['Purchase Price'] * df['Quantity']).sum()
                    asset_count = len(df)
                else:
                    total_value = 0
                    asset_count = len(df) if not df.empty else 0
            except:
                total_value = 0
                asset_count = 0
            
            timeline_data.append({
                'File': file,
                'Date': datetime.fromtimestamp(file_stats.st_mtime),
                'Total Value': total_value,
                'Asset Count': asset_count
            })
        except:
            continue
    
    if timeline_data:
        timeline_df = pd.DataFrame(timeline_data)
        timeline_df = timeline_df.sort_values('Date')
        
        fig = px.line(
            timeline_df,
            x='Date',
            y='Total Value',
            title="📊 Portfolio Value Timeline",
            markers=True,
            hover_data=['Asset Count', 'File']
        )
        fig.update_layout(
            yaxis_title="Total Portfolio Value ($)",
            xaxis_title="Date"
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# Help and Education Page
# ============================================================================

def help_page():
    """Enhanced help page with comprehensive guidance."""
    st.markdown(
        '<div class="main-header"><h1>❓ Help & Guide</h1><p>Learn how to maximize your investment management experience</p></div>',
        unsafe_allow_html=True
    )
    
    # Help navigation
    help_tab1, help_tab2, help_tab3, help_tab4 = st.tabs([
        "🚀 Getting Started",
        "📊 Understanding Metrics", 
        "🔧 Troubleshooting",
        "💡 Best Practices"
    ])
    
    with help_tab1:
        display_getting_started_help()
    
    with help_tab2:
        display_metrics_help()
    
    with help_tab3:
        display_troubleshooting_help()
    
    with help_tab4:
        display_best_practices_help()

def display_getting_started_help():
    """Display getting started guidance."""
    st.subheader("🚀 Getting Started")
    
    st.markdown("""
    ### Creating Your First Portfolio
    
    **Option 1: Add Assets Manually**
    1. Go to the **"➕ Add Asset"** tab
    2. Enter the ticker symbol (e.g., AAPL, MSFT, BTC-USD)
    3. Input purchase price, quantity, and asset type
    4. Click "Add Asset" to save
    
    **Option 2: Upload a File**
    1. Go to the **"📤 Upload Portfolio"** tab
    2. Download the CSV or JSON template
    3. Fill in your portfolio data
    4. Upload the file and import
    
    ### Required Information
    - **Ticker**: The trading symbol (AAPL, MSFT, etc.)
    - **Purchase Price**: What you paid per share/unit
    - **Quantity**: How many shares/units you own
    - **Asset Type**: Category (Stock, ETF, Crypto, etc.)
    """)
    
    with st.expander("📋 Sample Portfolio Format"):
        sample_data = pd.DataFrame({
            'Ticker': ['AAPL', 'MSFT', 'GOOGL'],
            'Purchase Price': [150.00, 300.00, 2500.00],
            'Quantity': [10, 5, 2],
            'Asset Type': ['Stock', 'Stock', 'Stock']
        })
        st.dataframe(sample_data)

def display_metrics_help():
    """Display metrics explanation."""
    st.subheader("📊 Understanding Key Metrics")
    
    metrics_info = {
        "💰 **Total Value**": "Current market value of your holdings (Current Price × Quantity)",
        "📈 **P/L (Profit/Loss)**": "Difference between current value and what you paid",
        "📊 **P/L %**": "Percentage return on your investment",
        "⚖️ **Weight %**": "Percentage of total portfolio value",
        "🎯 **RSI (Relative Strength Index)**": "Momentum indicator (0-100). Below 30 = oversold, above 70 = overbought",
        "📊 **Volatility**": "Annual price volatility percentage. Higher = more risky",
        "🔵 **Beta**": "Correlation with market. >1 = more volatile than market, <1 = less volatile",
        "🟢 **Alpha**": "Excess return vs benchmark. Positive = outperforming market",
        "📈 **Sharpe Ratio**": "Risk-adjusted return. Higher is better",
        "⚠️ **VaR (Value at Risk)**": "Potential loss at 95% confidence level"
    }
    
    for metric, description in metrics_info.items():
        st.markdown(f"- {metric}: {description}")
    
    if st.session_state.education_mode:
        st.info("💡 Education Mode is ON - you'll see helpful tooltips throughout the app!")

def display_troubleshooting_help():
    """Display troubleshooting information."""
    st.subheader("🔧 Common Issues & Solutions")
    
    issues = {
        "❌ Ticker not found": [
            "Double-check the ticker symbol spelling",
            "For crypto, use format like BTC-USD, ETH-USD",
            "International stocks may need exchange suffix",
            "Some delisted stocks won't have current prices"
        ],
        "⚠️ File upload errors": [
            "Ensure your file has required columns: Ticker, Purchase Price, Quantity, Asset Type",
            "Check that numeric columns contain valid numbers",
            "Remove any completely empty rows",
            "Save as UTF-8 encoding if using special characters"
        ],
        "📊 Missing data": [
            "Some metrics require historical data which may not be available",
            "New listings might not have enough price history",
            "Market closed - some data may be delayed",
            "Try refreshing data or checking your internet connection"
        ],
        "🔄 Slow performance": [
            "Clear browser cache if pages load slowly",
            "Large portfolios (>100 assets) may take longer to process",
            "Use 'Refresh Data' button to update cached prices",
            "Consider splitting very large portfolios"
        ]
    }
    
    for issue, solutions in issues.items():
        with st.expander(issue):
            for solution in solutions:
                st.write(f"• {solution}")

def display_best_practices_help():
    """Display best practices guidance."""
    st.subheader("💡 Best Practices")
    
    st.markdown("""
    ### Portfolio Management
    - **Diversify**: Don't put all money in one asset type
    - **Regular Review**: Check your portfolio at least monthly
    - **Keep Records**: Save notes about why you bought each asset
    - **Risk Management**: Don't risk more than you can afford to lose
    
    ### Using This App
    - **Education Mode**: Keep it on to learn about metrics
    - **Save Regularly**: Your portfolios auto-save, but manual saves create backups
    - **Historical Data**: Review portfolio history to track your progress
    - **Validate Tickers**: Use the validation feature when uploading files
    
    ### Investment Principles
    - **Long-term Focus**: Don't panic over short-term volatility
    - **Dollar-Cost Averaging**: Consider regular, consistent investments
    - **Rebalancing**: Periodically adjust allocations to target percentages
    - **Research**: Use the metrics as starting points, not final decisions
    """)
    
    st.warning("⚠️ **Disclaimer**: This app is for informational purposes only. Not financial advice. Always consult professionals for investment decisions.")

# ============================================================================
# Authentication Pages
# ============================================================================

def display_auth_page():
    """Enhanced authentication page with better UX."""
    # Header
    auth_header_html = """
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="background: linear-gradient(90deg, #3b82f6 0%, #10b981 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 3rem; margin-bottom: 1rem;">
            📊 Portfolio Manager Pro
        </h1>
        <p style="font-size: 1.3rem; color: #64748b; margin-bottom: 2rem;">
            Your comprehensive investment dashboard with real-time analytics
        </p>
    </div>
    """
    st.markdown(auth_header_html, unsafe_allow_html=True)
    
    # Feature highlights
    display_feature_highlights()
    
    st.markdown("---")
    
    # Authentication tabs
    tab1, tab2 = st.tabs(["🔐 Sign In", "📝 Create Account"])
    
    with tab1:
        display_login_form()
    
    with tab2:
        display_registration_form()
    
    # Security notice
    display_security_notice()

def display_feature_highlights():
    """Display app feature highlights."""
    col1, col2, col3 = st.columns(3)
    
    features = [
        ("📈 Real-Time Analytics", "Live market data with advanced metrics like Alpha, Beta, RSI, and Volatility"),
        ("📊 Interactive Dashboards", "Beautiful visualizations for portfolio allocation, performance, and risk analysis"), 
        ("🎯 Smart Recommendations", "AI-powered insights for diversification and portfolio optimization")
    ]
    
    for i, (title, description) in enumerate(features):
        with [col1, col2, col3][i]:
            feature_html = f"""
            <div style="text-align: center; padding: 1.5rem; background: white; border-radius: 15px; box-shadow: 0 4px 16px rgba(0,0,0,0.1); margin: 1rem 0; border: 1px solid #e2e8f0; height: 200px; display: flex; flex-direction: column; justify-content: center;">
                <h3 style="color: #1e293b; margin-bottom: 1rem;">{title}</h3>
                <p style="color: #64748b; margin: 0;">{description}</p>
            </div>
            """
            st.markdown(feature_html, unsafe_allow_html=True)

def display_security_notice():
    """Display security information."""
    security_html = """
    <div style="text-align: center; margin-top: 2rem; padding: 1rem; background-color: #f8fafc; border-radius: 10px; border: 1px solid #e2e8f0;">
        <small style="color: #64748b;">
            🔒 <strong>Your data is secure:</strong> Passwords are encrypted with PBKDF2-SHA256 • 
            All portfolio data is stored locally • No personal information is shared
        </small>
    </div>
    """
    st.markdown(security_html, unsafe_allow_html=True)

def display_login_form():
    """Enhanced login form with better validation."""
    st.markdown("### 🔐 Welcome Back!")
    st.write("Access your portfolio dashboard")
    
    with st.form("login_form"):
        username_input = st.text_input(
            "👤 Username",
            placeholder="Enter your username",
            help="The username you registered with"
        )
        
        password_input = st.text_input(
            "🔒 Password",
            type="password",
            placeholder="Enter your password",
            help="Your secure password"
        )
        
        remember_me = st.checkbox("🔄 Keep me signed in", help="Stay logged in for this session")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            submitted = st.form_submit_button("🚀 Sign In", type="primary")
        
        if submitted:
            handle_login_submission(username_input, password_input)

def handle_login_submission(username_input: str, password_input: str):
    """Handle login form submission."""
    if not username_input.strip():
        st.error("❌ Please enter your username")
    elif not password_input:
        st.error("❌ Please enter your password")
    else:
        with st.spinner("🔍 Verifying credentials..."):
            time.sleep(0.5)  # Brief delay for UX
            
            if authenticate_user(username_input.strip(), password_input):
                # Successful login
                st.session_state.authenticated = True
                st.session_state.username = username_input.strip()
                st.session_state.first_login = True
                st.session_state.show_welcome = True
                
                # Load user's portfolio
                safe_load_portfolio(username_input.strip())
                
                st.success("✅ Welcome back! Redirecting to your dashboard...")
                logger.info(f"User logged in: {username_input.strip()}")
                time.sleep(1)
                st.rerun()
            else:
                st.error("❌ Invalid username or password")
                
                if st.session_state.education_mode:
                    with st.expander("🔧 Login Help"):
                        st.markdown("""
                        **Trouble signing in?**
                        - Double-check your username and password
                        - Make sure Caps Lock is off
                        - Username is case-sensitive
                        - Contact support if you forgot your credentials
                        """)

def display_registration_form():
    """Enhanced registration form with validation."""
    st.markdown("### 📝 Join Portfolio Manager Pro")
    st.write("Create your account to start tracking investments")
    
    with st.form("register_form"):
        new_username = st.text_input(
            "👤 Choose Username",
            placeholder="Enter a unique username",
            help="3-20 characters, letters and numbers only"
        )
        
        new_password = st.text_input(
            "🔒 Create Password",
            type="password",
            placeholder="Minimum 6 characters",
            help="Use a strong password with letters, numbers, and symbols"
        )
        
        confirm_password = st.text_input(
            "🔒 Confirm Password",
            type="password",
            placeholder="Re-enter your password",
            help="Must match the password above"
        )
        
        # Password strength indicator
        if new_password:
            strength = putils.check_password_strength(new_password)
            strength_colors = {"Weak": "🔴", "Medium": "🟡", "Strong": "🟢"}
            st.write(f"Password Strength: {strength_colors.get(strength, '⚪')} {strength}")
        
        agree_terms = st.checkbox(
            "✅ I agree to the Terms of Service and Privacy Policy",
            help="Required to create an account"
        )
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            submitted_reg = st.form_submit_button("✨ Create Account", type="primary")
        
        if submitted_reg:
            handle_registration_submission(new_username, new_password, confirm_password, agree_terms)

def handle_registration_submission(new_username: str, new_password: str, confirm_password: str, agree_terms: bool):
    """Handle registration form submission."""
    # Validation
    errors = []
    username_clean = new_username.strip()
    
    if not username_clean:
        errors.append("Username is required")
    elif len(username_clean) < 3:
        errors.append("Username must be at least 3 characters")
    elif len(username_clean) > 20:
        errors.append("Username must be less than 20 characters")
    elif not username_clean.replace('_', '').isalnum():
        errors.append("Username can only contain letters, numbers, and underscores")
    
    if not new_password:
        errors.append("Password is required")
    elif len(new_password) < 6:
        errors.append("Password must be at least 6 characters")
    elif new_password != confirm_password:
        errors.append("Passwords do not match")
    
    if not agree_terms:
        errors.append("You must agree to the Terms of Service")
    
    if errors:
        for error in errors:
            st.error(f"❌ {error}")
    else:
        with st.spinner("👤 Creating your account..."):
            time.sleep(0.5)  # Brief delay for UX
            
            if register_user(username_clean, new_password):
                st.success("🎉 Account created successfully!")
                st.info("👆 You can now sign in using the Sign In tab")
                st.balloons()
                logger.info(f"New user registered: {username_clean}")
            else:
                st.error("❌ Username already exists. Please choose another.")

# ============================================================================
# Sidebar and Navigation
# ============================================================================

def create_sidebar():
    """Enhanced sidebar with user info and controls."""
    with st.sidebar:
        if st.session_state.authenticated:
            # User profile section
            profile_html = f"""
            <div class="sidebar-section">
                <h3>👤 Welcome Back!</h3>
                <p><strong>{st.session_state.username}</strong></p>
                <small>Last login: {datetime.now().strftime('%Y-%m-%d %H:%M')}</small>
            </div>
            """
            st.markdown(profile_html, unsafe_allow_html=True)
            
            # Portfolio quick stats
            display_sidebar_portfolio_stats()
            
            # Navigation
            st.markdown("### 🧭 Navigation")
            page = st.radio(
                "Choose a page:",
                [
                    "📊 Dashboard",
                    "➕ Add Asset",
                    "📤 Upload Portfolio",
                    "📚 Portfolio History",
                    "❓ Help",
                    "🚪 Sign Out"
                ],
                label_visibility="collapsed"
            )
            
            st.markdown("---")
            
            # Settings
            display_sidebar_settings()
            
            # Quick actions
            display_sidebar_quick_actions()
            
            # Footer
            display_sidebar_footer()
            
            return page
        
        else:
            # Unauthenticated sidebar
            display_unauthenticated_sidebar()
            return None

def display_sidebar_portfolio_stats():
    """Display portfolio quick stats in sidebar."""
    if st.session_state.portfolio_df is not None and not st.session_state.portfolio_df.empty:
        df = st.session_state.portfolio_df
        asset_count = len(df)
        
        if 'Purchase Price' in df.columns and 'Quantity' in df.columns:
            total_cost = (df['Purchase Price'] * df['Quantity']).sum()
        else:
            total_cost = 0
        
        stats_html = f"""
        <div class="sidebar-section">
            <h4>📊 Portfolio Quick Stats</h4>
            <p>🎯 <strong>{asset_count}</strong> assets</p>
            <p>💰 <strong>${total_cost:,.0f}</strong> invested</p>
            <p>📈 <strong>{df['Asset Type'].nunique()}</strong> asset types</p>
        </div>
        """
        st.markdown(stats_html, unsafe_allow_html=True)

def display_sidebar_settings():
    """Display settings section in sidebar."""
    st.markdown("### ⚙️ Settings")
    
    st.session_state.education_mode = st.checkbox(
        "📚 Education Mode",
        value=st.session_state.education_mode,
        help="Show helpful tooltips and explanations"
    )
    
    timeframe = st.selectbox(
        "📅 Data Timeframe",
        ["1mo", "3mo", "6mo", "1y", "2y"],
        index=2,
        help="Historical data period for analysis"
    )
    st.session_state.selected_timeframe = timeframe

def display_sidebar_quick_actions():
    """Display quick actions section in sidebar."""
    st.markdown("### ⚡ Quick Actions")
    
    if st.button("🔄 Refresh All Data", help="Update all market data"):
        clear_cache()
        st.success("✅ Data refreshed!")
        st.rerun()
    
    if st.session_state.portfolio_df is not None and not st.session_state.portfolio_df.empty:
        if st.button("💾 Save Current Portfolio", help="Save current state"):
            try:
                putils.save_portfolio(st.session_state.username, st.session_state.portfolio_df)
                st.success("✅ Portfolio saved!")
            except Exception as e:
                st.error(f"❌ Save failed: {e}")

def display_sidebar_footer():
    """Display sidebar footer."""
    st.markdown("---")
    footer_html = f"""
    <div style="text-align: center; color: #64748b; font-size: 0.8rem;">
        <p>📊 Portfolio Manager Pro v{st.session_state.app_version}</p>
        <p>Built with ❤️ using Streamlit</p>
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)

def display_unauthenticated_sidebar():
    """Display sidebar for unauthenticated users."""
    signin_html = """
    <div class="sidebar-section">
        <h3>🔐 Please Sign In</h3>
        <p>Access your portfolio dashboard by signing in or creating an account.</p>
    </div>
    """
    st.markdown(signin_html, unsafe_allow_html=True)
    
    st.markdown("### 🌟 Features")
    st.markdown("""
    - 📈 **Real-time market data**
    - 📊 **Interactive charts**
    - 🎯 **Risk analysis**
    - 💡 **Smart recommendations**
    - 📱 **Mobile responsive**
    - 🔒 **Secure & private**
    """)

# ============================================================================
# Logout Functionality
# ============================================================================

def display_logout_confirmation():
    """Enhanced logout confirmation."""
    st.markdown(
        '<div class="main-header"><h1>🚪 Sign Out</h1><p>Thanks for using Portfolio Manager Pro!</p></div>',
        unsafe_allow_html=True
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        logout_html = f"""
        <div style="text-align: center; padding: 2rem; background: white; border-radius: 15px; box-shadow: 0 4px 16px rgba(0,0,0,0.1); border: 1px solid #e2e8f0;">
            <h3 style="color: #1e293b; margin-bottom: 1rem;">👋 See you soon, {st.session_state.username}!</h3>
            <p style="color: #64748b; margin-bottom: 1rem;">Your portfolio data has been saved securely.</p>
            <p style="color: #64748b;">You can return anytime to continue tracking your investments.</p>
        </div>
        """
        st.markdown(logout_html, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            if st.button("⬅️ Stay Signed In", type="secondary"):
                st.info("👍 Continuing your session...")
                time.sleep(1)
                st.rerun()
        
        with col_b:
            if st.button("🚪 Confirm Sign Out", type="primary"):
                handle_logout()

def handle_logout():
    """Handle user logout process."""
    username = st.session_state.username
    
    # Clear session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
    # Reinitialize
    initialize_session_state()
    
    st.success("👋 You have been signed out successfully!")
    logger.info(f"User logged out: {username}")
    time.sleep(1)
    st.rerun()

# ============================================================================
# Main Application Logic
# ============================================================================

def main():
    """Enhanced main application with improved error handling."""
    try:
        # Create sidebar and get navigation choice
        selected_page = create_sidebar()
        
        if not st.session_state.authenticated:
            display_auth_page()
            return
        
        # Show welcome message for new sessions
        show_welcome_message()
        
        # Main content routing
        route_to_page(selected_page)
        
    except Exception as e:
        handle_application_error(e)

def route_to_page(selected_page: str):
    """Route to the selected page."""
    if selected_page == "📊 Dashboard":
        display_portfolio_overview()
    elif selected_page == "➕ Add Asset":
        add_asset_page()
    elif selected_page == "📤 Upload Portfolio":
        upload_portfolio_page()
    elif selected_page == "📚 Portfolio History":
        history_page()
    elif selected_page == "❓ Help":
        help_page()
    elif selected_page == "🚪 Sign Out":
        display_logout_confirmation()

def handle_application_error(e: Exception):
    """Handle application-level errors."""
    error_msg = f"An unexpected error occurred: {str(e)}"
    st.error(f"❌ {error_msg}")
    logger.error(f"Application error: {e}", exc_info=True)
    
    if st.session_state.education_mode:
        with st.expander("🔧 Error Details (for debugging)"):
            st.code(traceback.format_exc())
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("🔄 Refresh Page"):
            st.rerun()
    
    with col2:
        if st.button("🏠 Go to Dashboard"):
            # Reset to dashboard
            st.session_state.show_welcome = False
            st.rerun()
    
    with col3:
        if st.button("🆘 Reset Application"):
            # Clear all session state and restart
            for key in list(st.session_state.keys()):
                if key not in ['authenticated', 'username']:  # Keep login status
                    del st.session_state[key]
            initialize_session_state()
            st.session_state.authenticated = True  # Restore auth status
            st.rerun()

# ============================================================================
# Application Entry Point
# ============================================================================

if __name__ == "__main__":
    main()
