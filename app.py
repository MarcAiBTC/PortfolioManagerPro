"""
Enhanced Financial Portfolio Manager - Main Application
=====================================================

A comprehensive Streamlit application for managing investment portfolios with 
advanced visualizations, real-time metrics, and intelligent analysis.

Features:
- Secure user authentication with PBKDF2 encryption
- Interactive portfolio management and visualization
- Real-time market data integration via Yahoo Finance
- Advanced financial metrics (Alpha, Beta, RSI, Volatility)
- Educational tooltips and explanations throughout
- Responsive design with modern UI/UX

Author: Enhanced by AI Assistant
"""

import os
import time
import traceback
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from portfolio_utils import PORTFOLIO_DIR

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from auth import authenticate_user, register_user
import portfolio_utils as putils

# ============================================================================
# Configuration and Setup
# ============================================================================

st.set_page_config(
    page_title="📊 Portfolio Manager Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS styling
st.markdown("""
<style>
    /* Main styling */
    .main-header {
        background: linear-gradient(90deg, #1f77b4 0%, #ff7f0e 100%);
        color: white;
        text-align: center;
        padding: 2rem 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        margin: 0.5rem 0;
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
    }
    
    .welcome-banner {
        background: linear-gradient(135deg, #e6f3ff 0%, #f0f8ff 100%);
        border: 2px solid #1f77b4;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(31,119,180,0.1);
    }
    
    .success-badge {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        margin: 0.25rem;
        font-size: 0.9rem;
        box-shadow: 0 2px 8px rgba(76,175,80,0.3);
    }
    
    .warning-badge {
        background: linear-gradient(135deg, #ff9800 0%, #f57c00 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        margin: 0.25rem;
        font-size: 0.9rem;
        box-shadow: 0 2px 8px rgba(255,152,0,0.3);
    }
    
    .info-tooltip {
        background-color: #e7f3ff;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .performance-positive {
        color: #4CAF50;
        font-weight: bold;
    }
    
    .performance-negative {
        color: #f44336;
        font-weight: bold;
    }
    
    .sidebar-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    
    .chart-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

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
        'selected_timeframe': '6mo'
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

initialize_session_state()

# ============================================================================
# UI Helper Functions
# ============================================================================

def show_tooltip(text: str, tooltip: str):
    """Display text with a tooltip."""
    return f"{text} ℹ️" if st.session_state.education_mode else text

def show_welcome_message():
    """Enhanced welcome message with onboarding guidance."""
    if st.session_state.show_welcome and st.session_state.authenticated:
        st.markdown(f"""
        <div class="welcome-banner">
            <h2>🎉 Welcome to Portfolio Manager Pro, {st.session_state.username}!</h2>
            <p><strong>Your comprehensive investment dashboard is ready!</strong></p>

           col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 What you can do:")
    st.markdown("""
    - 📈 **Track performance** with real-time data
    - 📋 **Add assets** manually or upload CSV/JSON
    - 🎯 **Analyze risk** with Alpha, Beta, RSI metrics
    - 📊 **Visualize allocation** with interactive charts
    """)

with col2:
    st.subheader("🚀 Quick Start:")
    st.markdown("""
    1. Add some assets or upload a portfolio
    2. Explore the interactive dashboards
    3. Use tooltips (ℹ️) to learn about metrics
    4. Check diversification recommendations
    """)

# Pro tip box
st.info("💡 **Pro Tip:** Enable Education Mode in the sidebar to see helpful explanations throughout the app!")
        
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

def create_metric_card(title: str, value: str, delta: str = None, help_text: str = None):
    """Create a styled metric card with optional delta and help."""
    delta_class = "performance-positive" if delta and not delta.startswith("-") else "performance-negative"
    delta_html = f'<div class="{delta_class}">{delta}</div>' if delta else ""
    help_html = f'<small style="opacity: 0.8;">{help_text}</small>' if help_text else ""
    
    return f"""
    <div class="metric-card">
        <h3 style="margin: 0; font-size: 1.1rem;">{title}</h3>
        <div style="font-size: 1.8rem; font-weight: bold; margin: 0.5rem 0;">{value}</div>
        {delta_html}
        {help_html}
    </div>
    """

def load_and_set_portfolio(username: str, filename: Optional[str] = None) -> bool:
    """Enhanced portfolio loading with better error handling."""
    try:
        with st.spinner("📂 Loading portfolio..."):
            df = putils.load_portfolio(username, filename)
            
        if df is not None and not df.empty:
            st.session_state.portfolio_df = df
            st.session_state.selected_portfolio_file = filename
            st.session_state.portfolio_modified = False
            st.session_state.last_refresh = datetime.now()
            
            st.success(f"✅ Portfolio loaded successfully! ({len(df)} assets)")
            return True
        else:
            st.warning("⚠️ Portfolio file is empty or could not be loaded")
            return False
            
    except Exception as e:
        st.error(f"❌ Error loading portfolio: {str(e)}")
        if st.session_state.education_mode:
            with st.expander("🔍 Troubleshooting Help"):
                st.write("""
                **Common issues:**
                - File format not supported (use CSV or JSON)
                - Missing required columns (Ticker, Purchase Price, Quantity, Asset Type)
                - Invalid data types in numeric columns
                - Corrupted file
                """)
        return False

# ============================================================================
# Main Dashboard Functions
# ============================================================================

def display_portfolio_overview():
    """Enhanced portfolio overview with comprehensive visualizations."""
    st.markdown('<div class="main-header"><h1>📊 Portfolio Dashboard</h1><p>Real-time analysis of your investments</p></div>', unsafe_allow_html=True)
    
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
                    load_and_set_portfolio(username, selected_file)
            
            with col2:
                if st.button("🔄 Refresh Data", help="Update prices and recalculate metrics"):
                    st.session_state.price_cache = {}
                    st.session_state.price_cache_time = 0
                    if st.session_state.portfolio_df is not None:
                        st.rerun()
            
            with col3:
                if st.button("📊 Quick Stats", help="Show portfolio summary"):
                    if st.session_state.portfolio_df is not None:
                        df = st.session_state.portfolio_df
                        st.info(f"Assets: {len(df)} | Types: {df['Asset Type'].nunique()} | Last updated: {st.session_state.last_refresh.strftime('%H:%M') if st.session_state.last_refresh else 'Unknown'}")
        else:
            st.info("📝 No portfolios found. Create your first portfolio by adding assets or uploading a file!")

    df = st.session_state.portfolio_df
    if df is None or df.empty:
        display_empty_portfolio_guide()
        return

    # Fetch current data with enhanced error handling
    try:
        with st.spinner("📡 Fetching real-time market data..."):
            tickers = df['Ticker'].tolist()
            price_dict = putils.get_cached_prices(tickers)
            
            # Get benchmark data for advanced metrics
            benchmark_data = putils.fetch_benchmark_data()
            st.session_state.benchmark_data = benchmark_data
            
            metrics_df = putils.compute_enhanced_metrics(df, price_dict, benchmark_data)
            
            # Check for failed price fetches
            failed_tickers = [t for t, p in price_dict.items() if pd.isna(p)]
            if failed_tickers:
                st.warning(f"⚠️ Could not fetch prices for: {', '.join(failed_tickers)}")
                if st.session_state.education_mode:
                    with st.expander("💡 Why might this happen?"):
                        st.write("""
                        - Ticker symbol might be incorrect or delisted
                        - Market is closed and no recent data available
                        - Network connectivity issues
                        - Yahoo Finance API limitations
                        """)
    
    except Exception as e:
        st.error(f"❌ Error fetching market data: {str(e)}")
        return

    # Enhanced Portfolio Summary Metrics
    display_portfolio_summary(metrics_df)
    
    # Main Dashboard Tabs
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

def display_empty_portfolio_guide():
    """Guide for users with empty portfolios."""
    st.markdown("""
    <div style="text-align: center; padding: 3rem; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 15px; margin: 2rem 0;">
        <h2>🚀 Let's Build Your Portfolio!</h2>
        <p style="font-size: 1.2rem; margin-bottom: 2rem;">Start tracking your investments with our comprehensive tools</p>
        
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 2rem; margin: 2rem 0;">
            <div style="background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                <h3>➕ Add Assets Manually</h3>
                <p>Start by adding individual stocks, ETFs, crypto, or other assets one by one</p>
            </div>
            <div style="background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                <h3>📤 Upload Portfolio</h3>
                <p>Import your existing portfolio from CSV or JSON files</p>
            </div>
            <div style="background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                <h3>📚 Learn as You Go</h3>
                <p>Use Education Mode to understand metrics and make better decisions</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

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
        st.markdown(create_metric_card(
            f"{pl_symbol} Total P/L",
            f"${total_pl:,.2f}",
            f"{total_pl_pct:+.2f}%",
            help_text="Profit/Loss vs purchase price"
        ), unsafe_allow_html=True)
    
    with col3:
        best_performer = metrics_df.loc[metrics_df['P/L %'].idxmax(), 'Ticker'] if not metrics_df['P/L %'].isna().all() else "N/A"
        best_pl = metrics_df['P/L %'].max() if not metrics_df['P/L %'].isna().all() else 0
        st.markdown(create_metric_card(
            "🏆 Best Performer",
            best_performer,
            f"+{best_pl:.1f}%",
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

def display_performance_analysis(metrics_df: pd.DataFrame):
    """Enhanced performance analysis with multiple chart types."""
    st.subheader("📊 Performance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # P/L Distribution Chart
        if not metrics_df.empty:
            fig = px.bar(
                metrics_df.nlargest(10, 'P/L'),
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
        if not metrics_df.empty:
            fig = px.bar(
                metrics_df.nlargest(10, 'P/L %'),
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
    
    # Risk vs Return Scatter Plot
    if 'Alpha' in metrics_df.columns and 'Beta' in metrics_df.columns:
        st.subheader("🎯 Risk vs Return Analysis")
        
        fig = px.scatter(
            metrics_df,
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
                - Positive: Outperforming the market
                - Negative: Underperforming the market
                
                **Beta (X-axis):** Measures volatility vs market
                - Beta > 1: More volatile than market
                - Beta < 1: Less volatile than market
                - Beta = 1: Moves with market
                
                **Ideal Quadrant:** High Alpha, Low Beta (top-left)
                """)

def display_allocation_analysis(metrics_df: pd.DataFrame):
    """Asset allocation visualizations."""
    st.subheader("🥧 Asset Allocation Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Asset Type Distribution
        allocation_by_type = metrics_df.groupby('Asset Type')['Total Value'].sum().reset_index()
        
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
    st.subheader("🎯 Portfolio Concentration")
    
    # Calculate concentration metrics
    total_value = metrics_df['Total Value'].sum()
    top_5_concentration = metrics_df.nlargest(5, 'Total Value')['Total Value'].sum() / total_value * 100
    top_10_concentration = metrics_df.nlargest(10, 'Total Value')['Total Value'].sum() / total_value * 100
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Top 5 Holdings", f"{top_5_concentration:.1f}%", 
                 help="Percentage of portfolio in top 5 holdings")
    
    with col2:
        st.metric("Top 10 Holdings", f"{top_10_concentration:.1f}%",
                 help="Percentage of portfolio in top 10 holdings")
    
    with col3:
        herfindahl_index = ((metrics_df['Weight %'] / 100) ** 2).sum()
        concentration_level = "High" if herfindahl_index > 0.25 else "Medium" if herfindahl_index > 0.15 else "Low"
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
    
    # Risk Distribution Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Volatility Distribution
        if 'Volatility' in metrics_df.columns:
            fig = px.histogram(
                metrics_df,
                x='Volatility',
                nbins=20,
                title="📊 Volatility Distribution",
                labels={'Volatility': 'Volatility (%)', 'count': 'Number of Assets'},
                color_discrete_sequence=['#1f77b4']
            )
            fig.add_vline(x=metrics_df['Volatility'].mean(), line_dash="dash", 
                         line_color="red", annotation_text="Average")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Beta Distribution
        if 'Beta' in metrics_df.columns:
            fig = px.histogram(
                metrics_df,
                x='Beta',
                nbins=20,
                title="📊 Beta Distribution",
                labels={'Beta': 'Beta (Market Risk)', 'count': 'Number of Assets'},
                color_discrete_sequence=['#ff7f0e']
            )
            fig.add_vline(x=1, line_dash="dash", line_color="gray", 
                         annotation_text="Market Beta")
            fig.add_vline(x=metrics_df['Beta'].mean(), line_dash="dash", 
                         line_color="red", annotation_text="Portfolio Avg")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Risk Heatmap
    st.subheader("🔥 Risk Heatmap")
    
    risk_metrics = ['P/L %', 'Beta', 'Volatility', 'RSI']
    available_metrics = [col for col in risk_metrics if col in metrics_df.columns]
    
    if available_metrics:
        risk_data = metrics_df[['Ticker'] + available_metrics].set_index('Ticker')
        
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
                - 🟢 Green: Lower risk/better performance
                - 🟡 Yellow: Medium risk
                - 🔴 Red: Higher risk/worse performance
                
                **What to look for:**
                - Assets with many red cells need attention
                - Diversification across green/yellow is healthy
                - Extreme values (very red/green) indicate outliers
                """)

def display_holdings_detail(metrics_df: pd.DataFrame):
    """Detailed holdings table with enhanced formatting."""
    st.subheader("📋 Detailed Holdings")
    
    # Formatting functions
    def format_currency(val):
        return f"${val:,.2f}" if pd.notna(val) else "N/A"
    
    def format_percentage(val):
        return f"{val:.2f}%" if pd.notna(val) else "N/A"
    
    def format_number(val, decimals=2):
        return f"{val:.{decimals}f}" if pd.notna(val) else "N/A"
    
    # Create display dataframe
    display_df = metrics_df.copy()
    
    # Format columns for display
    format_columns = {
        'Purchase Price': format_currency,
        'Current Price': format_currency,
        'Total Value': format_currency,
        'Cost Basis': format_currency,
        'P/L': format_currency,
        'P/L %': format_percentage,
        'Weight %': format_percentage,
        'Alpha': lambda x: format_number(x, 3),
        'Beta': lambda x: format_number(x, 2),
        'RSI': lambda x: format_number(x, 1),
        'Volatility': format_percentage
    }
    
    for col, formatter in format_columns.items():
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(formatter)
    
    # Color coding function
    def highlight_performance(val):
        """Color code performance metrics."""
        if 'N/A' in str(val):
            return 'color: gray'
        try:
            if '%' in str(val):
                num_val = float(str(val).replace('%', '').replace('€', '').replace('.', '').replace(',', '.').strip())
                if num_val > 10:
                    return 'background-color: #d4edda; color: #155724; font-weight: bold'
                elif num_val > 0:
                    return 'background-color: #d1ecf1; color: #0c5460'
                elif num_val < -10:
                    return 'background-color: #f8d7da; color: #721c24; font-weight: bold'
                elif num_val < 0:
                    return 'background-color: #f5c6cb; color: #721c24'
            return ''
        except:
            return ''
    
    # Apply styling
    styled_df = display_df.style.applymap(
        highlight_performance, 
        subset=[col for col in ['P/L %', 'P/L'] if col in display_df.columns]
    )
    
    # Display options
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        search_term = st.text_input("🔍 Search assets:", placeholder="Enter ticker or asset type...")
    
    with col2:
        sort_by = st.selectbox("📊 Sort by:", 
                              ['Total Value', 'P/L %', 'P/L', 'Ticker', 'Weight %'])
    
    with col3:
        ascending = st.checkbox("Ascending order", value=False)
    
    # Filter and sort
    filtered_df = display_df.copy()
    if search_term:
        mask = filtered_df['Ticker'].str.contains(search_term, case=False, na=False) | \
               filtered_df['Asset Type'].str.contains(search_term, case=False, na=False)
        filtered_df = filtered_df[mask]
    
    if sort_by in metrics_df.columns:
        sort_values = metrics_df[sort_by] if sort_by in metrics_df.columns else filtered_df[sort_by]
        filtered_df = filtered_df.loc[sort_values.sort_values(ascending=ascending).index]
    
    # Display the table
    st.dataframe(
        filtered_df.style.applymap(highlight_performance, 
                                  subset=[col for col in ['P/L %', 'P/L'] if col in filtered_df.columns]),
        use_container_width=True,
        height=400
    )
    
    # Export options
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
        # Generate PDF report (placeholder for future implementation)
        if st.button("📊 Generate Report", help="Create detailed PDF report"):
            st.info("📊 PDF report generation coming soon!")

def display_recommendations(metrics_df: pd.DataFrame):
    """Intelligent portfolio recommendations."""
    st.subheader("🎯 Portfolio Recommendations")
    
    recommendations = putils.generate_portfolio_recommendations(metrics_df)
    
    if recommendations:
        for i, rec in enumerate(recommendations):
            rec_type = rec.get('type', 'info')
            icon = {"warning": "⚠️", "success": "✅", "info": "💡"}.get(rec_type, "📌")
            
            st.markdown(f"""
            <div class="{'warning-badge' if rec_type == 'warning' else 'success-badge' if rec_type == 'success' else 'info-tooltip'}">
                {icon} <strong>{rec['title']}</strong><br>
                {rec['description']}
            </div>
            """, unsafe_allow_html=True)
    
    # Rebalancing suggestions
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
            st.plotly_chart(current_fig, use_container_width=True)
        
        with col2:
            st.write("**Suggested Allocation:**")
            suggested_fig = px.pie(
                rebalancing_data['suggested'],
                values='weight',
                names='asset_type',
                title="Suggested Distribution"
            )
            st.plotly_chart(suggested_fig, use_container_width=True)


# ==========================================================================
# Help and Onboarding Page
# ==========================================================================

def help_page():
    """Display a comprehensive help and onboarding page.

    This page provides contextual assistance and explains how to use the
    portfolio manager. It includes examples of required file formats,
    guidance on interpreting metrics, and tips for maintaining a well
    diversified portfolio.  It is intended to be accessible to both
    beginners and advanced users.
    """
    st.markdown('<div class="main-header"><h1>❓ Ayuda y Guía</h1><p>Aprende a sacar el máximo partido a tu gestor de inversiones</p></div>', unsafe_allow_html=True)

    st.subheader("📁 Importar y crear carteras")
    st.markdown("""
    Puedes crear una cartera de dos maneras:

    1. **Añadir activos manualmente** desde la pestaña **"➕ Add Asset"**. Introduce el ticker, el precio de compra, la cantidad y la categoría del activo. Obtendrás una vista previa inmediata con el coste y el valor estimado.
    2. **Subir un archivo CSV o JSON** desde la pestaña **"📤 Upload Portfolio"**. El archivo debe contener al menos las columnas: *Ticker*, *Purchase Price*, *Quantity* y *Asset Type*. Puedes descargar una plantilla de ejemplo desde esa misma página.
    
    Ten en cuenta que los tickers deben ser válidos según Yahoo Finance. Utiliza el botón **"Validate Tickers"** tras cargar tu archivo para verificar su validez.
    """
    )

    st.subheader("📊 Interpretar los análisis y métricas")
    st.markdown("""
    Una vez tengas una cartera cargada, la pestaña **"📊 Dashboard"** te mostrará una visión general con diversos análisis:

    - **Performance Analysis:** gráficos de barras que muestran tus mejores y peores inversiones y un scatter de **Alpha** vs **Beta** para comparar rentabilidad y riesgo relativo al mercado.
    - **Asset Allocation Analysis:** tartas que desglosan tu cartera por tipo de activo y por las mayores posiciones, junto con indicadores de concentración (Índice Herfindahl).
    - **Risk Analysis:** un panel con la beta media, la volatilidad, el ratio Sharpe y el VaR (Value at Risk) del conjunto, además de histogramas y un heatmap para evaluar el riesgo individual de cada activo.
    - **Detailed Holdings:** una tabla interactiva donde puedes buscar y ordenar por cualquier métrica. Los valores positivos/negativos se resaltan para facilitar la lectura.
    - **Recommendations:** sugerencias automáticas basadas en la diversificación, volatilidad, RSI y resultados. Aquí también encontrarás una propuesta de rebalanceo.

    **Consejo:** activa el modo de educación en la barra lateral para que aparezcan explicaciones y tooltips (ℹ️) en distintos puntos de la aplicación.
    """
    )

    st.subheader("📈 Comprender los indicadores clave")
    st.markdown("""
    - **RSI (Relative Strength Index):** oscila entre 0 y 100. Valores por debajo de 30 indican posible **sobreventa** (oportunidad de compra), mientras que valores por encima de 70 indican **sobrecompra** (posible corrección).
    - **Volatilidad:** mide la variación de los precios. Una volatilidad anualizada superior al 40% indica mayor riesgo.
    - **Beta:** compara la volatilidad del activo con el mercado (S&P 500 o NASDAQ). *Beta > 1* implica que el activo es más volátil que el mercado; *Beta < 1* implica que es menos volátil.
    - **Alpha:** mide el rendimiento en exceso frente al mercado. Valores positivos significan que el activo está superando al índice de referencia.
    - **Sharpe Ratio:** ratio que relaciona el exceso de rentabilidad con la volatilidad. Cuanto mayor sea, mejor relación rentabilidad/riesgo.
    - **Value at Risk (VaR):** estima la pérdida máxima que podrías soportar con un nivel de confianza determinado (p. ej. 95%).
    """
    )

    st.subheader("🎯 Recomendaciones y buenas prácticas")
    st.markdown("""
    - **Diversifica:** no concentres tu cartera en una sola clase de activo. Apunta a que tus principales cinco posiciones no excedan el 50% del total.
    - **Revisa regularmente:** actualiza los precios haciendo clic en **"🔄 Refresh All Data"** desde la barra lateral para mantener las métricas al día.
    - **Valora tu horizonte temporal:** los indicadores técnicos como el RSI o la volatilidad pueden variar según el plazo considerado. Puedes ajustar el periodo histórico en la barra lateral (*Data Timeframe*).
    - **Guarda tus cambios:** cada vez que añadas activos o subas un nuevo fichero, la cartera se guarda automáticamente. Aun así, puedes pulsar **"Save Current Portfolio"** para asegurarte.
    - **Consulta la historia:** en la sección **"📚 Portfolio History"** puedes revisar versiones anteriores de tus carteras y descargar cualquier copia.
    
    Si tienes dudas adicionales, puedes contactar con soporte o consultar la documentación incluida en el repositorio.
    """
    )

    st.info("ℹ️ Esta aplicación está diseñada tanto para principiantes como para inversores avanzados. Usa las funcionalidades que más se adapten a tu nivel y recuerda que ninguna recomendación sustituye el asesoramiento financiero profesional.")

# ============================================================================
# Asset Management Functions
# ============================================================================

def add_asset_page():
    """Enhanced asset addition with improved UX and validation."""
    st.markdown('<div class="main-header"><h1>➕ Add New Asset</h1><p>Expand your portfolio with new investments</p></div>', unsafe_allow_html=True)
    
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
        
        # Form submission
        col1, col2 = st.columns([1, 3])
        
        with col1:
            submitted = st.form_submit_button("➕ Add Asset", type="primary")
        
        with col2:
            if submitted:
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
                else:
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
                        
                    except Exception as e:
                        st.error(f"❌ Error adding asset: {str(e)}")

def upload_portfolio_page():
    """Enhanced portfolio upload with better validation and preview."""
    st.markdown('<div class="main-header"><h1>📤 Upload Portfolio</h1><p>Import your existing investment data</p></div>', unsafe_allow_html=True)
    
    username = st.session_state.username
    
    # File format guide
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
    
    # File upload section
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
        try:
            # Parse the file
            with st.spinner("📖 Reading file..."):
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:  # JSON
                    df = pd.read_json(uploaded_file)
            
            st.success(f"✅ File '{uploaded_file.name}' loaded successfully!")
            
            # Validation
            required_cols = {'Ticker', 'Purchase Price', 'Quantity', 'Asset Type'}
            missing_cols = required_cols - set(df.columns)
            
            if missing_cols:
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                return
            
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
            
            if df.empty:
                st.error("❌ No valid data remaining after cleaning")
                return
            
            # Preview section
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
                avg_position_size = total_cost / len(df)
                st.metric("📈 Avg Position", f"${avg_position_size:,.2f}")
            
            # Data table preview
            st.dataframe(df, use_container_width=True, height=300)
            
            # Asset type breakdown
            if len(df) > 0:
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
            
            # Confirmation
            st.subheader("💾 Confirm Import")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                if st.button("🚀 Import Portfolio", type="primary"):
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
                        
                    except Exception as e:
                        st.error(f"❌ Error saving portfolio: {str(e)}")
            
            with col2:
                if st.button("🔍 Validate Tickers", help="Check if all tickers are valid"):
                    with st.spinner("🔍 Validating ticker symbols..."):
                        tickers = df['Ticker'].unique().tolist()
                        validation_results = putils.validate_tickers(tickers)
                        
                        valid_tickers = [t for t, valid in validation_results.items() if valid]
                        invalid_tickers = [t for t, valid in validation_results.items() if not valid]
                        
                        if invalid_tickers:
                            st.warning(f"⚠️ Invalid tickers found: {', '.join(invalid_tickers)}")
                        else:
                            st.success("✅ All tickers validated successfully!")
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            
            if st.session_state.education_mode:
                with st.expander("🔧 Troubleshooting Tips"):
                    st.markdown("""
                    **Common Issues:**
                    1. **File Format**: Ensure your file is a valid CSV or JSON
                    2. **Column Names**: Check that column names match exactly
                    3. **Data Types**: Ensure prices and quantities are numbers
                    4. **Encoding**: Try saving your file with UTF-8 encoding
                    5. **Empty Cells**: Remove any completely empty rows
                    """)

def history_page():
    """Enhanced portfolio history management."""
    st.markdown('<div class="main-header"><h1>📚 Portfolio History</h1><p>Manage your saved portfolios</p></div>', unsafe_allow_html=True)
    
    username = st.session_state.username
    files = putils.list_portfolios(username)
    
    if not files:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 15px; margin: 2rem 0;">
            <h2>📝 No Portfolio History Yet</h2>
            <p style="font-size: 1.2rem; margin-bottom: 2rem;">Start building your investment tracking history!</p>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin: 2rem 0;">
                <div style="background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                    <h4>➕ Add Assets</h4>
                    <p>Start by adding individual investments</p>
                </div>
                <div style="background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                    <h4>📤 Upload Files</h4>
                    <p>Import from CSV or JSON files</p>
                </div>
                <div style="background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                    <h4>💾 Auto-Save</h4>
                    <p>Your portfolios are saved automatically</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return
    
    st.write(f"📊 You have **{len(files)}** saved portfolios:")
    
    # Portfolio management interface
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
                if load_and_set_portfolio(username, selected_file):
                    st.success(f"✅ Loaded '{selected_file}'")
        
        with col_b:
            if st.button("🗑️ Delete", help="Delete selected portfolio"):
                if selected_file == st.session_state.selected_portfolio_file:
                    st.error("❌ Cannot delete currently active portfolio")
                elif st.button("⚠️ Confirm Delete", type="secondary"):
                    try:
                        file_path = os.path.join(putils.PORTFOLIO_DIR, selected_file)
                        os.remove(file_path)
                        st.success(f"✅ Deleted '{selected_file}'")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error deleting file: {e}")
    
    if selected_file:
        # File details and preview
        st.subheader(f"📄 Portfolio Details: {selected_file}")
        
        try:
            file_path = os.path.join(putils.PORTFOLIO_DIR, selected_file)
            
            if os.path.exists(file_path):
                file_stats = os.stat(file_path)
                file_size = file_stats.st_size
                file_modified = datetime.fromtimestamp(file_stats.st_mtime)
                
                # File metadata
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
                with st.expander("👀 Portfolio Preview", expanded=True):
                    try:
                        preview_df = pd.read_csv(file_path) if selected_file.endswith('.csv') else pd.read_json(file_path)
                        
                        if not preview_df.empty:
                            # Quick stats
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("🎯 Assets", len(preview_df))
                            
                            with col2:
                                total_cost = (preview_df['Purchase Price'] * preview_df['Quantity']).sum()
                                st.metric("💰 Total Cost", f"${total_cost:,.2f}")
                            
                            with col3:
                                asset_types = preview_df['Asset Type'].nunique()
                                st.metric("📊 Asset Types", asset_types)
                            
                            # Data preview
                            st.dataframe(preview_df, use_container_width=True, height=200)
                            
                            # Download option
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
                        else:
                            st.warning("⚠️ Portfolio file appears to be empty")
                    
                    except Exception as preview_error:
                        st.error(f"❌ Error loading preview: {preview_error}")
        
        except Exception as detail_error:
            st.error(f"❌ Error loading file details: {detail_error}")
    
    # Portfolio timeline visualization
    if len(files) > 1:
        st.subheader("📈 Portfolio Timeline")
        
        timeline_data = []
        for file in files:
            try:
                file_path = os.path.join(putils.PORTFOLIO_DIR, file)
                file_stats = os.stat(file_path)
                
                # Try to load and calculate value
                try:
                    df = pd.read_csv(file_path) if file.endswith('.csv') else pd.read_json(file_path)
                    total_value = (df['Purchase Price'] * df['Quantity']).sum()
                    asset_count = len(df)
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
# Authentication Functions
# ============================================================================

def display_auth_page():
    """Enhanced authentication page with better UX."""
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="background: linear-gradient(90deg, #1f77b4 0%, #ff7f0e 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 3rem;">
            📊 Portfolio Manager Pro
        </h1>
        <p style="font-size: 1.3rem; color: #666; margin-bottom: 2rem;">
            Your comprehensive investment dashboard with real-time analytics
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature highlights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem; background: white; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); margin: 1rem 0;">
            <h3>📈 Real-Time Analytics</h3>
            <p>Live market data with advanced metrics like Alpha, Beta, RSI, and Volatility</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem; background: white; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); margin: 1rem 0;">
            <h3>📊 Interactive Dashboards</h3>
            <p>Beautiful visualizations for portfolio allocation, performance, and risk analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem; background: white; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); margin: 1rem 0;">
            <h3>🎯 Smart Recommendations</h3>
            <p>AI-powered insights for diversification and portfolio optimization</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Authentication tabs
    tab1, tab2 = st.tabs(["🔐 Sign In", "📝 Create Account"])
    
    with tab1:
        display_login_form()
    
    with tab2:
        display_registration_form()
    
    # Security notice
    st.markdown("""
    <div style="text-align: center; margin-top: 2rem; padding: 1rem; background-color: #f8f9fa; border-radius: 10px;">
        <small>
            🔒 <strong>Your data is secure:</strong> Passwords are encrypted with PBKDF2-SHA256 • 
            All portfolio data is stored locally • No personal information is shared
        </small>
    </div>
    """, unsafe_allow_html=True)

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
                        load_and_set_portfolio(username_input.strip())
                        
                        st.success("✅ Welcome back! Redirecting to your dashboard...")
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
            strength_color = {"Weak": "🔴", "Medium": "🟡", "Strong": "🟢"}
            st.write(f"Password Strength: {strength_color.get(strength, '⚪')} {strength}")
        
        agree_terms = st.checkbox(
            "✅ I agree to the Terms of Service and Privacy Policy",
            help="Required to create an account"
        )
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            submitted_reg = st.form_submit_button("✨ Create Account", type="primary")
        
        if submitted_reg:
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
                    else:
                        st.error("❌ Username already exists. Please choose another.")

# ============================================================================
# Main Application Logic
# ============================================================================

def create_sidebar():
    """Enhanced sidebar with user info and controls."""
    with st.sidebar:
        if st.session_state.authenticated:
            # User profile section
            st.markdown(f"""
            <div class="sidebar-section">
                <h3>👤 Welcome Back!</h3>
                <p><strong>{st.session_state.username}</strong></p>
                <small>Last login: {datetime.now().strftime('%Y-%m-%d %H:%M')}</small>
            </div>
            """, unsafe_allow_html=True)
            
            # Portfolio quick stats
            if st.session_state.portfolio_df is not None and not st.session_state.portfolio_df.empty:
                df = st.session_state.portfolio_df
                asset_count = len(df)
                total_cost = (df['Purchase Price'] * df['Quantity']).sum()
                
                st.markdown(f"""
                <div class="sidebar-section">
                    <h4>📊 Portfolio Quick Stats</h4>
                    <p>🎯 <strong>{asset_count}</strong> assets</p>
                    <p>💰 <strong>${total_cost:,.0f}</strong> invested</p>
                    <p>📈 <strong>{df['Asset Type'].nunique()}</strong> asset types</p>
                </div>
                """, unsafe_allow_html=True)
            
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
            
            # Quick actions
            st.markdown("### ⚡ Quick Actions")
            
            if st.button("🔄 Refresh All Data", help="Update all market data"):
                st.session_state.price_cache = {}
                st.session_state.price_cache_time = 0
                st.session_state.benchmark_data = None
                st.success("✅ Data refreshed!")
                st.rerun()
            
            if st.session_state.portfolio_df is not None and not st.session_state.portfolio_df.empty:
                if st.button("💾 Save Current Portfolio", help="Save current state"):
                    try:
                        putils.save_portfolio(st.session_state.username, st.session_state.portfolio_df)
                        st.success("✅ Portfolio saved!")
                    except Exception as e:
                        st.error(f"❌ Save failed: {e}")
            
            # Footer
            st.markdown("---")
            st.markdown("""
            <div style="text-align: center; color: #666; font-size: 0.8rem;">
                <p>📊 Portfolio Manager Pro v2.0</p>
                <p>Built with ❤️ using Streamlit</p>
            </div>
            """, unsafe_allow_html=True)
            
            return page
        
        else:
            # Unauthenticated sidebar
            st.markdown("""
            <div class="sidebar-section">
                <h3>🔐 Please Sign In</h3>
                <p>Access your portfolio dashboard by signing in or creating an account.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 🌟 Features")
            st.markdown("""
            - 📈 **Real-time market data**
            - 📊 **Interactive charts**
            - 🎯 **Risk analysis**
            - 💡 **Smart recommendations**
            - 📱 **Mobile responsive**
            - 🔒 **Secure & private**
            """)
            
            return None

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
        
    except Exception as e:
        st.error(f"❌ An unexpected error occurred: {str(e)}")
        
        if st.session_state.education_mode:
            with st.expander("🔧 Error Details (for debugging)"):
                st.code(traceback.format_exc())
        
        if st.button("🔄 Restart Application"):
            st.session_state.clear()
            st.rerun()

def display_logout_confirmation():
    """Enhanced logout confirmation."""
    st.markdown('<div class="main-header"><h1>🚪 Sign Out</h1><p>Thanks for using Portfolio Manager Pro!</p></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 2rem; background: white; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
            <h3>👋 See you soon, {username}!</h3>
            <p>Your portfolio data has been saved securely.</p>
            <p>You can return anytime to continue tracking your investments.</p>
        </div>
        """.format(username=st.session_state.username), unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            if st.button("⬅️ Stay Signed In", type="secondary"):
                st.info("👍 Continuing your session...")
                time.sleep(1)
                st.rerun()
        
        with col_b:
            if st.button("🚪 Confirm Sign Out", type="primary"):
                # Clear session state
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                
                # Reinitialize
                initialize_session_state()
                
                st.success("👋 You have been signed out successfully!")
                time.sleep(1)
                st.rerun()

if __name__ == "__main__":
    main()
