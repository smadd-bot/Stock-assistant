import streamlit as st
import yfinance as yf
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from datetime import datetime, timedelta
import re

# --- Page Configuration ---
st.set_page_config(
    page_title="Breakout Detection Dashboard",
    page_icon="📈",
    layout="wide"
)

# --- Professional Disclaimer ---
st.warning("⚠️ **DISCLAIMER:** This dashboard is not financial advice and is provided for educational purposes only. Please do your own research before making any trading or investment decisions.")
st.markdown("<h3 style='text-align: center; color: #4CAF50;'>🧘‍♂️ BREATHE! Patience, React, Don't Predict!</h3>", unsafe_allow_html=True)

# --- App Constants ---
WATCHLIST = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META']


# --- Data Caching ---
@st.cache_data(ttl="15m")
def get_stock_data(ticker_symbol):
    """Fetches stock data (info and history) from yfinance."""
    ticker = yf.Ticker(ticker_symbol)
    info = ticker.info
    # Fetch 1 year of historical data for analysis
    hist = ticker.history(period="1y")
    if hist.empty:
        return None, None
    return info, hist

@st.cache_data(ttl="15m")
def get_options_data(ticker_symbol):
    """Fetches options chain data (calls and puts) for the nearest expiry."""
    ticker = yf.Ticker(ticker_symbol)
    try:
        # Get the nearest expiration date
        nearest_expiry = ticker.options[0]
        opts = ticker.option_chain(nearest_expiry)
        # Return only the serializable DataFrames
        return opts.calls, opts.puts
    except IndexError:
        # Ticker might not have options
        return None, None

# --- Logic Layer ---

SECTOR_ETF_MAP = {
    'Communication Services': 'XLC',
    'Consumer Cyclical': 'XLY',
    'Consumer Defensive': 'XLP',
    'Energy': 'XLE',
    'Financial Services': 'XLF',
    'Healthcare': 'XLV',
    'Industrials': 'XLI',
    'Real Estate': 'XLRE',
    'Technology': 'XLK',
    'Utilities': 'XLU',
    'Basic Materials': 'XLB'
}

def format_option_symbol(symbol):
    """Translates an option symbol like AAPL260401C00252500 into a human-readable format."""
    match = re.match(r'^([A-Z]+)(\d{2})(\d{2})(\d{2})([CP])(\d{8})$', symbol)
    if match:
        ticker, yy, mm, dd, opt_type, strike_str = match.groups()
        strike = float(strike_str) / 1000
        return f"{ticker} ${strike:g} {opt_type} (20{yy}-{mm}-{dd})"
    return symbol

def analyze_options_flow(calls, puts):
    """
    Calculates unusual volume and Put/Call ratio as a proxy for options flow.
    Unusual Volume: Daily Volume > Open Interest.
    """
    if calls is None or puts is None or calls.empty or puts.empty:
        return 0, 0, 0, pd.DataFrame()


    # Filter for unusual volume
    unusual_calls = calls[calls['volume'] > calls['openInterest']].copy()
    unusual_puts = puts[puts['volume'] > puts['openInterest']].copy()

    unusual_call_volume = unusual_calls['volume'].sum()
    unusual_put_volume = unusual_puts['volume'].sum()

    unusual_calls['Type'] = 'Call'
    unusual_puts['Type'] = 'Put'
    
    unusual_calls['Sentiment'] = 'Bullish'
    unusual_puts['Sentiment'] = 'Bearish'

    # Calculate Money Moved (Premium = Volume * Last Price * 100)
    unusual_calls['Premium'] = unusual_calls['volume'] * unusual_calls['lastPrice'] * 100
    unusual_puts['Premium'] = unusual_puts['volume'] * unusual_puts['lastPrice'] * 100

    # Extract Timestamp
    if 'lastTradeDate' in unusual_calls.columns:
        unusual_calls['Time'] = pd.to_datetime(unusual_calls['lastTradeDate']).dt.strftime('%m-%d %H:%M')
    else:
        unusual_calls['Time'] = 'N/A'
        
    if 'lastTradeDate' in unusual_puts.columns:
        unusual_puts['Time'] = pd.to_datetime(unusual_puts['lastTradeDate']).dt.strftime('%m-%d %H:%M')
    else:
        unusual_puts['Time'] = 'N/A'

    # Avoid division by zero
    if unusual_call_volume > 0:
        pc_ratio = unusual_put_volume / unusual_call_volume
    else:
        pc_ratio = 0

    # Combine for display
    unusual_activity_df = pd.concat([unusual_calls, unusual_puts], ignore_index=True)
    if not unusual_activity_df.empty:
        unusual_activity_df['Contract'] = unusual_activity_df['contractSymbol'].apply(format_option_symbol)
        
    return unusual_call_volume, unusual_put_volume, pc_ratio, unusual_activity_df

def calculate_rsi(series, period=14):
    """Calculates the Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

@st.cache_data(ttl="1h")
def get_popularity_trend(ticker_symbol):
    """Fetches Google Trends data for the ticker symbol over the last 90 days."""
    try:
        from pytrends.request import TrendReq
        # Removed retries and backoff_factor to fix urllib3 v2+ compatibility issue (method_whitelist)
        pytrends = TrendReq(hl='en-US', tz=360)
        kw_list = [f"{ticker_symbol} stock"]
        pytrends.build_payload(kw_list, cat=0, timeframe='today 3-m', geo='US', gprop='')
        df = pytrends.interest_over_time()
        return df[kw_list[0]] if not df.empty and kw_list[0] in df.columns else None
    except Exception as e:
        return str(e) # Return error message gracefully to handle rate limits

def analyze_sector_strength(stock_info, stock_hist):
    """Compares stock RSI and 30-day performance against its sector ETF."""
    sector = stock_info.get('sector')
    if not sector or sector not in SECTOR_ETF_MAP:
        return None

    sector_etf_symbol = SECTOR_ETF_MAP[sector]
    sector_etf_hist = yf.Ticker(sector_etf_symbol).history(period="1y")

    if sector_etf_hist.empty:
        return None

    # Calculate RSI (using custom pandas function)
    stock_hist['RSI'] = calculate_rsi(stock_hist['Close'], period=14)
    sector_etf_hist['RSI'] = calculate_rsi(sector_etf_hist['Close'], period=14)

    stock_rsi = stock_hist['RSI'].iloc[-1]
    sector_rsi = sector_etf_hist['RSI'].iloc[-1]

    # Calculate 30-day relative strength
    end_date = stock_hist.index[-1]
    start_date = end_date - timedelta(days=30)
    
    stock_30d = stock_hist['Close'].truncate(before=start_date)
    sector_30d = sector_etf_hist['Close'].truncate(before=start_date)

    stock_perf = (stock_30d.iloc[-1] / stock_30d.iloc[0]) - 1
    sector_perf = (sector_30d.iloc[-1] / sector_30d.iloc[0]) - 1
    
    relative_strength = stock_perf - sector_perf

    return {
        "sector": sector,
        "etf": sector_etf_symbol,
        "stock_rsi": stock_rsi,
        "sector_rsi": sector_rsi,
        "relative_strength_30d": relative_strength
    }

def evaluate_trade_setup(stock_hist, sector_analysis, stock_info):
    """
    Evaluates a strict checklist for a breakout trade setup and returns a confidence score.
    """
    if stock_hist.empty or len(stock_hist) < 150:
        return None, 0, None

    current_price = stock_hist['Close'].iloc[-1]
    current_vol = stock_hist['Volume'].iloc[-1]
    
    ma50 = stock_hist['MA50'].iloc[-1]
    ma150 = stock_hist['MA150'].iloc[-1]
    vol_ma20 = stock_hist['Vol_MA20'].iloc[-1]

    past_50 = stock_hist['Close'][:-1].tail(50)
    resistance_50d = past_50.max()
    
    cr = stock_info.get('currentRatio')
    de = stock_info.get('debtToEquity')
    roe = stock_info.get('returnOnEquity')

    # 1. Checklist Conditions
    checklist = {
        "Strong Profitability (ROE > 15%)": roe is not None and roe >= 0.15,
        "Healthy Liquidity (Current Ratio > 1.2)": cr is not None and cr >= 1.2,
        "Manageable Debt (Debt to Equity < 100%)": de is not None and de <= 100,
        "Price broke out above 50-Day Resistance": current_price > resistance_50d,
        "Price is above 50-Day Moving Average": current_price > ma50,
        "Price is above 150-Day Moving Average": current_price > ma150,
        "Significant Volume (Today's Vol > 1.5x 20-Day Avg)": current_vol > (vol_ma20 * 1.5),
        "Sector Tailwind (Stock Momentum > Sector ETF)": sector_analysis['stock_rsi'] > sector_analysis['sector_rsi'] if sector_analysis else False
    }

    # 2. Confidence Score Calculation
    score = sum(checklist.values())
    confidence = (score / len(checklist)) * 100

    # 3. Trade Setup (Only generated if 100% confident based on strict rules)
    setup = None
    if confidence == 100:
        entry_price = current_price
        # Stop loss placed safely below the 50-day MA
        stop_loss = ma50 if ma50 < current_price else current_price * 0.95
        risk = entry_price - stop_loss
        # Take Profit mapped to a 1:2 Risk/Reward ratio
        take_profit = entry_price + (risk * 2)
        
        setup = {
            "entry": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit
        }

    return checklist, confidence, setup

def generate_swot(info, hist, sector_analysis):
    """Generates a heuristic rule-based SWOT analysis (max 3 bullets per category)."""
    swot = {'Strengths': [], 'Weaknesses': [], 'Opportunities': [], 'Threats': []}

    # Strengths
    if info.get('returnOnEquity', 0) and info.get('returnOnEquity', 0) > 0.15: swot['Strengths'].append("Strong Return on Equity (>15%)")
    if info.get('profitMargins', 0) and info.get('profitMargins', 0) > 0.10: swot['Strengths'].append("Healthy Profit Margins (>10%)")
    if hist['Close'].iloc[-1] > hist['MA150'].iloc[-1]: swot['Strengths'].append("Long-term price trend is bullish (Price > 150D MA)")

    # Weaknesses
    if info.get('debtToEquity', 0) and info.get('debtToEquity', 0) > 150: swot['Weaknesses'].append("High Debt-to-Equity ratio (>150%)")
    if info.get('currentRatio', 2) and info.get('currentRatio', 2) < 1.0: swot['Weaknesses'].append("Poor short-term liquidity (Current Ratio < 1.0)")
    if info.get('revenueGrowth', 1) and info.get('revenueGrowth', 1) < 0: swot['Weaknesses'].append("Negative recent revenue growth")

    # Opportunities
    if sector_analysis and sector_analysis['stock_rsi'] > sector_analysis['sector_rsi']: swot['Opportunities'].append(f"Outperforming its sector ({sector_analysis['sector']})")
    if hist['Volume'].iloc[-1] > hist['Vol_MA20'].iloc[-1] * 1.5: swot['Opportunities'].append("Recent surge in trading volume indicates high interest")
    if info.get('targetMeanPrice') and info.get('targetMeanPrice') > hist['Close'].iloc[-1] * 1.1: swot['Opportunities'].append("Analyst mean price target implies >10% upside")

    # Threats
    if info.get('shortPercentOfFloat', 0) and info.get('shortPercentOfFloat', 0) > 0.05: swot['Threats'].append("High short interest (>5% of float)")
    if hist['Close'].iloc[-1] < hist['MA50'].iloc[-1]: swot['Threats'].append("Short-term price trend is bearish (Price < 50D MA)")
    if sector_analysis and sector_analysis['stock_rsi'] < 40: swot['Threats'].append("Stock momentum is currently weak (RSI < 40)")

    # Ensure defaults and limit to 3 bullets
    for k in swot.keys():
        swot[k] = swot[k][:3] if swot[k] else ["Neutral indicator / No extreme signals detected."]
        
    return swot

def plot_chart(df, ticker_symbol):
    """Creates an interactive Plotly chart with Price, MAs, Volume, and Volume Profile."""
    fig = make_subplots(
        rows=2, cols=2,
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.03,
        horizontal_spacing=0.0,
        column_widths=[0.85, 0.15],
        row_heights=[0.8, 0.2],
        specs=[[{}, {}],
               [{}, None]]
    )

    # Find missing dates to perfectly remove gaps (weekends, holidays)
    idx = df.index
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    idx = idx.normalize()
    all_dates = pd.date_range(start=idx.min(), end=idx.max())
    missing_dates = all_dates.difference(idx).strftime('%Y-%m-%d').tolist()

    # --- Row 1, Col 1: Price Chart ---
    # Candlestick
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'], name='Price'),
                  row=1, col=1)

    # 20-Day Moving Average
    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='MA20',
                             line=dict(color='skyblue', width=1.5)),
                  row=1, col=1)

    # 50-Day Moving Average
    fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='MA50',
                             line=dict(color='orange', width=1.5)),
                  row=1, col=1)
                  
    # 150-Day Moving Average
    fig.add_trace(go.Scatter(x=df.index, y=df['MA150'], name='MA150',
                             line=dict(color='purple', width=1.5, dash='dot')),
                  row=1, col=1)

    # 50-day Support & Resistance Lines
    past_50 = df['Close'][:-1].tail(50)
    resistance_level = past_50.max()
    support_level = past_50.min()
    
    fig.add_hline(y=resistance_level, line_dash="dash", line_color="red",
                  annotation_text=f"50-Day Res: {resistance_level:.2f}",
                  annotation_position="bottom right", row=1, col=1)
    fig.add_hline(y=support_level, line_dash="dash", line_color="green",
                  annotation_text=f"50-Day Sup: {support_level:.2f}",
                  annotation_position="bottom right", row=1, col=1)

    # --- Row 1, Col 2: Volume Profile ---
    fig.add_trace(go.Histogram(
        y=df['Close'],
        x=df['Volume'],
        histfunc='sum',
        name='Volume Profile',
        orientation='h',
        marker_color='rgba(150,150,150,0.4)',
        nbinsy=100
    ), row=1, col=2)

    # --- Row 2, Col 1: Volume by Time ---
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color='rgba(100,100,100,0.5)'),
                  row=2, col=1)

    # --- Layout Updates ---
    fig.update_layout(
        title_text=f'{ticker_symbol} Daily Chart with Volume Profile',
        xaxis_rangeslider_visible=False,
        showlegend=False,
        hovermode='x unified', # Enables the vertical hover line
        margin=dict(l=20, r=20, t=40, b=20),
        height=800 # Increased height to make the chart more square
    )
    # Apply rangebreaks to flawlessly hide weekends and holidays without breaking Candlestick
    fig.update_xaxes(
        rangebreaks=[dict(values=missing_dates)],
        showticklabels=False, # Hides x-axis dates, reliant on hover for cleaner UI
        col=1 # Crucial: Only apply to the time-series charts, not the Volume Profile
    )

    # Update axes titles and visibility
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(showticklabels=False, row=1, col=2) # Hide y-axis labels on profile
    fig.update_xaxes(showticklabels=False, title_text="", row=1, col=2) # Hide x-axis on profile

    return fig

# --- UI/UX ---
st.title("Breakout Detection & Flow Analysis Dashboard")

# --- Sidebar for Inputs ---
st.sidebar.header("Stock Selection")

# Initialize watchlist in session state if it doesn't exist
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = WATCHLIST.copy()

# CSV Uploader
uploaded_file = st.sidebar.file_uploader(
    "Upload Custom Watchlist (CSV)",
    type="csv",
    help="Upload a CSV file with ticker symbols in the first column."
)

if uploaded_file is not None:
    try:
        # To prevent reprocessing the same file on every rerun
        if uploaded_file.name != st.session_state.get('last_uploaded_filename'):
            df = pd.read_csv(uploaded_file)
            if not df.empty:
                # Get tickers from the first column, clean them up
                new_tickers = df.iloc[:, 0].dropna().astype(str).str.strip().str.upper().tolist()
                # Combine with default, remove duplicates, sort
                st.session_state.watchlist = sorted(list(set(WATCHLIST + new_tickers)))
                st.session_state.last_uploaded_filename = uploaded_file.name
                st.sidebar.success(f"Watchlist updated with {len(new_tickers)} tickers.")
                # Use st.rerun() to ensure the selectbox is updated immediately with the new list
                st.rerun()
    except Exception as e:
        st.sidebar.error(f"Error processing CSV: {e}")
        # Reset to avoid trying to re-process a bad file
        st.session_state.last_uploaded_filename = None

# Ticker Input using tabs for clarity
tab_watchlist, tab_single = st.sidebar.tabs(["Watchlist", "Single Ticker"])

with tab_watchlist:
    selected_ticker = st.selectbox("Select from Watchlist", st.session_state.watchlist)

with tab_single:
    typed_ticker = st.text_input("Or Enter a Ticker Symbol").upper()

# Priority logic: if user typed a ticker, use it. Otherwise, use the one from the watchlist.
if typed_ticker:
    ticker_symbol = typed_ticker
else:
    ticker_symbol = selected_ticker

tab_breakout, tab_scalp, tab_heatmap = st.tabs(["Breakout/Down", "Potential Morning Scalp", "Heat Map of all Stocks"])

with tab_breakout:
    if ticker_symbol:
        # Fetch Data
        info, hist = get_stock_data(ticker_symbol)
        
        if hist is None:
            st.error(f"Could not fetch data for '{ticker_symbol}'. Please check the ticker.")
        else:
            # --- Calculate Indicators ---
            hist['MA20'] = hist['Close'].rolling(window=20).mean()
            hist['MA50'] = hist['Close'].rolling(window=50).mean()
            hist['MA150'] = hist['Close'].rolling(window=150).mean()
            hist['Vol_MA20'] = hist['Volume'].rolling(window=20).mean()
            
            # --- Calculate Performance Metrics ---
            price_today = hist['Close'].iloc[-1]
            price_1d_ago = hist['Close'].iloc[-2] if len(hist['Close']) > 1 else price_today
            price_7d_ago = hist['Close'].iloc[-8] if len(hist['Close']) > 7 else None
            price_30d_ago = hist['Close'].iloc[-31] if len(hist['Close']) > 30 else None
            
            # 52-week high/low
            fifty_two_week_high = hist['High'].max()
            fifty_two_week_low = hist['Low'].min()

            # --- Run Analysis ---
            calls_df, puts_df = get_options_data(ticker_symbol)
            unusual_call_vol, unusual_put_vol, pc_ratio, unusual_activity_df = analyze_options_flow(calls_df, puts_df)
            sector_analysis = analyze_sector_strength(info, hist.copy()) # Pass a copy to avoid mutation issues
            checklist, confidence, setup = evaluate_trade_setup(hist, sector_analysis, info)

            # --- Display UI ---

            # Display Confidence & Setup
            if checklist:
                st.subheader(f"🧠 Setup Confidence Score: {confidence:.0f}%")
                
                # Checklist Display
                for condition, met in checklist.items():
                    icon = "✅" if met else "❌"
                    st.write(f"{icon} {condition}")

                # Actionable Trade Box
                if setup:
                    st.success(f"""
                    🎯 **HIGH CONFIDENCE TRADE SETUP DETECTED**
                    * **Get In Price:** ~${setup['entry']:.2f} (Market Open)
                    * **Stop Loss:** ${setup['stop_loss']:.2f} (Below 50D Support)
                    * **Take Profit:** ${setup['take_profit']:.2f} (1:2 Risk/Reward)
                    """)
            else:
                st.warning("Not enough data to calculate setup confidence (requires 150 trading days).")

            st.markdown("---")

            # Display the main chart
            st.plotly_chart(plot_chart(hist, ticker_symbol), use_container_width=True)

            # Main content in tabs below the chart
            perf_tab, opts_tab, sector_tab, swot_tab, pop_tab = st.tabs(["📈 Performance", "📊 Options Flow", "Sector Comparison", "🧩 SWOT Analysis", "🔥 Popularity"])

            with perf_tab:
                st.header("📈 Performance")
                st.caption(f"{info.get('shortName', ticker_symbol)} ({ticker_symbol}) | Sector: {info.get('sector', 'N/A')}")

                col1, col2, col3 = st.columns(3)
                with col1:
                    # Current Price with 1-day change
                    st.metric(
                        "Current Price",
                        f"${price_today:.2f}",
                        f"${price_today - price_1d_ago:.2f} ({((price_today/price_1d_ago)-1):.2%})" if price_1d_ago != 0 else "N/A"
                    )
                    # 7-Day Change
                    if price_7d_ago is not None:
                        st.metric("7-Day Change", f"{((price_today/price_7d_ago)-1):.2%}", f"from ${price_7d_ago:.2f}")
                    else:
                        st.metric("7-Day Change", "N/A")
                    # 30-Day Change
                    if price_30d_ago is not None:
                        st.metric("30-Day Change", f"{((price_today/price_30d_ago)-1):.2%}", f"from ${price_30d_ago:.2f}")
                    else:
                        st.metric("30-Day Change", "N/A")
                
                with col2:
                    st.metric("52-Week High", f"${fifty_two_week_high:.2f}")
                    st.metric("52-Week Low", f"${fifty_two_week_low:.2f}")
                    
                with col3:
                    st.write("Position in 52-Week Range:")
                    try:
                        price_range_pct = (price_today - fifty_two_week_low) / (fifty_two_week_high - fifty_two_week_low)
                        st.progress(price_range_pct)
                    except ZeroDivisionError:
                        st.progress(0.5) # Handle case where high and low are the same
                        
                st.markdown("---")
                st.subheader("🏦 Fundamental Health")
                f_col1, f_col2 = st.columns(2)
                f_col3, f_col4 = st.columns(2)
                
                cr = info.get('currentRatio')
                de = info.get('debtToEquity')
                roe = info.get('returnOnEquity')
                total_cash = info.get('totalCash')
                total_debt = info.get('totalDebt')
                
                with f_col1:
                    if cr is not None:
                        cr_rating = "🟢 Good" if cr >= 1.2 else ("🟡 Fair" if cr >= 1.0 else "🔴 Bad")
                        st.metric("Current Ratio", f"{cr:.2f}")
                        st.caption(f"{cr_rating}: Evaluates ability to pay short-term obligations. > 1.2 is generally healthy liquidity.")
                    else:
                        st.metric("Current Ratio", "N/A")
                with f_col2:
                    if de is not None:
                        de_rating = "🟢 Good" if de < 100 else ("🟡 Moderate" if de <= 200 else "🔴 High")
                        st.metric("Debt to Equity", f"{de:.2f}%")
                        st.caption(f"{de_rating}: Evaluates financial leverage. Lower percentage means less reliance on debt.")
                    else:
                        st.metric("Debt to Equity", "N/A")
                with f_col3:
                    if roe is not None:
                        roe_rating = "🟢 Good" if roe >= 0.15 else ("🟡 Fair" if roe > 0 else "🔴 Bad")
                        st.metric("Return on Equity", f"{roe*100:.2f}%")
                        st.caption(f"{roe_rating}: Measures profitability from shareholder equity. > 15% is strong.")
                    else:
                        st.metric("Return on Equity", "N/A")
                with f_col4:
                    if total_cash is not None and total_debt is not None:
                        if total_debt == 0:
                            st.metric("Cash to Debt Ratio", "No Debt")
                            st.caption("🟢 Excellent: Margin of safety. Company has cash and zero debt.")
                        else:
                            cd_ratio = total_cash / total_debt
                            cd_rating = "🟢 Good" if cd_ratio >= 0.75 else ("🟡 Fair" if cd_ratio >= 0.5 else "🔴 Risky")
                            st.metric("Cash to Debt Ratio", f"{cd_ratio:.2f}")
                            st.caption(f"{cd_rating}: Margin of safety. > 0.75 shows strong cash reserves vs debt.")
                    else:
                        st.metric("Cash to Debt Ratio", "N/A")
                        
                st.markdown("---")
                st.subheader("📊 Income Statement Health")
                i_col1, i_col2 = st.columns(2)
                i_col3, i_col4 = st.columns(2)
                
                rev_growth = info.get('revenueGrowth')
                earn_growth = info.get('earningsGrowth')
                gross_margin = info.get('grossMargins')
                op_margin = info.get('operatingMargins')
                
                with i_col1:
                    if rev_growth is not None:
                        rg_rating = "🟢 Excellent" if rev_growth >= 0.20 else ("🟡 Fair" if rev_growth > 0 else "🔴 Poor")
                        st.metric("Revenue Growth (YoY)", f"{rev_growth*100:.2f}%")
                        st.caption(f"{rg_rating}: Top-line sales growth.")
                    else:
                        st.metric("Revenue Growth", "N/A")
                with i_col2:
                    if earn_growth is not None:
                        eg_rating = "🟢 Excellent" if earn_growth >= 0.15 else ("🟡 Fair" if earn_growth > 0 else "🔴 Poor")
                        st.metric("Earnings Growth (YoY)", f"{earn_growth*100:.2f}%")
                        st.caption(f"{eg_rating}: Bottom-line EPS growth.")
                    else:
                        st.metric("Earnings Growth", "N/A")
                with i_col3:
                    if gross_margin is not None:
                        gm_rating = "🟢 Excellent" if gross_margin >= 0.50 else ("🟡 Fair" if gross_margin >= 0.30 else "🔴 Poor")
                        st.metric("Gross Margin", f"{gross_margin*100:.2f}%")
                        st.caption(f"{gm_rating}: Profitability after direct costs.")
                    else:
                        st.metric("Gross Margin", "N/A")
                with i_col4:
                    if op_margin is not None:
                        om_rating = "🟢 Excellent" if op_margin >= 0.15 else ("🟡 Fair" if op_margin > 0 else "🔴 Poor")
                        st.metric("Operating Margin", f"{op_margin*100:.2f}%")
                        st.caption(f"{om_rating}: Profitability after operating expenses.")
                    else:
                        st.metric("Operating Margin", "N/A")
                        
                # Rule of 40 calculation
                if rev_growth is not None and op_margin is not None:
                    rule_of_40 = (rev_growth + op_margin) * 100
                    r40_rating = "🟢 Elite (Passes Rule of 40)" if rule_of_40 >= 40 else "🟡 Does not pass Rule of 40"
                    st.write(f"**Rule of 40 Score:** {rule_of_40:.1f} ({r40_rating}) - *A popular Wall Street metric combining growth and profit margins.*")

            with opts_tab:
                st.header("📊 Options Flow Proxy")
                
                o_col1, o_col2, o_col3 = st.columns(3)
                with o_col1:
                    st.metric("Unusual Put/Call Ratio", f"{pc_ratio:.2f}")
                with o_col2:
                    st.metric("Unusual Call Volume", f"{unusual_call_vol:,}")
                with o_col3:
                    st.metric("Unusual Put Volume", f"{unusual_put_vol:,}")
                    
                st.info("Based on contracts where daily volume exceeds open interest.")
                
                st.markdown("---")
                st.subheader("Top Unusual Contracts Today")
                if not unusual_activity_df.empty:
                    display_cols = {
                        'Time': 'Time', 'Contract': 'Contract', 'Type': 'Type', 'Sentiment': 'Sentiment', 
                        'strike': 'Strike', 'volume': 'Volume', 'lastPrice': 'Last Price', 'Premium': 'Premium ($)'
                    }
                    display_df = unusual_activity_df[list(display_cols.keys())].rename(columns=display_cols)
                    
                    def color_sentiment(val):
                        color = '#00C851' if val == 'Bullish' else '#ff4444'
                        return f'color: {color}; font-weight: bold;'
                        
                    styled_df = display_df.sort_values('Volume', ascending=False).head(10).reset_index(drop=True).style\
                        .map(color_sentiment, subset=['Sentiment'])\
                        .format({'Premium ($)': '${:,.2f}'})
                    st.dataframe(styled_df, use_container_width=True, hide_index=True)
                    
                    # Summing Bullish vs Bearish Premium
                    bullish_prem = unusual_activity_df[unusual_activity_df['Sentiment'] == 'Bullish']['Premium'].sum()
                    bearish_prem = unusual_activity_df[unusual_activity_df['Sentiment'] == 'Bearish']['Premium'].sum()
                    
                    st.markdown("---")
                    bp_col1, bp_col2 = st.columns(2)
                    with bp_col1:
                        st.metric("🟢 Total Bullish Premium", f"${bullish_prem:,.2f}")
                    with bp_col2:
                        st.metric("🔴 Total Bearish Premium", f"${bearish_prem:,.2f}")
                else:
                    st.info("No unusual options activity detected today.")

            with sector_tab:
                st.header("Sector Comparison")
                if sector_analysis:
                    st.metric(f"Sector ETF ({sector_analysis['etf']})", f"{sector_analysis['sector']}")
                    st.metric("30-Day Relative Strength", f"{sector_analysis['relative_strength_30d']:.2%}", help="Stock 30d % change vs. Sector 30d % change")
                    st.metric("Stock RSI (14)", f"{sector_analysis['stock_rsi']:.2f}")
                    st.metric("Sector RSI (14)", f"{sector_analysis['sector_rsi']:.2f}")
                    st.markdown("""
                    ---
                    **💡 How to interpret this:**
                    * **Relative Strength > 0%** means the stock has outperformed its sector ETF over the last 30 days.
                    * **Stock RSI > Sector RSI** suggests the stock has stronger upward momentum than its sector peers right now.
                    """)
                else:
                    st.warning("Could not perform sector analysis.")
                    
            with swot_tab:
                st.header("🧩 Rule-Based SWOT Analysis")
                st.caption("Generated automatically from technical indicators and fundamental balance sheet data.")
                swot = generate_swot(info, hist, sector_analysis)

                col_s, col_w = st.columns(2)
                with col_s:
                    st.subheader("💪 Strengths")
                    for item in swot['Strengths']: st.markdown(f"- {item}")
                with col_w:
                    st.subheader("📉 Weaknesses")
                    for item in swot['Weaknesses']: st.markdown(f"- {item}")

                col_o, col_t = st.columns(2)
                with col_o:
                    st.subheader("🔭 Opportunities")
                    for item in swot['Opportunities']: st.markdown(f"- {item}")
                with col_t:
                    st.subheader("⚠️ Threats")
                    for item in swot['Threats']: st.markdown(f"- {item}")

            with pop_tab:
                st.header("📈 Google Trends Popularity")
                st.write("Measures relative search interest for the stock over the last 90 days.")
                trend_data = get_popularity_trend(ticker_symbol)
                if isinstance(trend_data, str):
                    st.warning(f"Google Trends is currently rate-limiting requests. Please try again later. (Error: {trend_data})")
                elif trend_data is not None and not trend_data.empty:
                    fig_trend = go.Figure(data=go.Scatter(x=trend_data.index, y=trend_data.values, mode='lines', line=dict(color='magenta', width=2)))
                    fig_trend.update_layout(title=f"US Search Interest: '{ticker_symbol} stock'", xaxis_title="Date", yaxis_title="Interest (0-100)", height=400, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig_trend, use_container_width=True)
                else:
                    st.info("Not enough search data available for this ticker.")

with tab_scalp:
    st.header("Potential Morning Scalp")
    st.info("Morning scalp detection logic and metrics will be displayed here.")
    st.markdown("*(This feature is in the backlog)*")

with tab_heatmap:
    st.header("Heat Map of all Stocks")
    st.info("Market heat map visualization will be displayed here.")
    st.markdown("*(This feature is in the backlog)*")