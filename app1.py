import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import numpy as np
import ta
from textblob import TextBlob
import matplotlib.pyplot as plt
import google.generativeai as genai
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
import datetime
from functools import lru_cache
import asyncio
import aiohttp
import random
from concurrent.futures import ThreadPoolExecutor
from fallback_stocks import FALLBACK_STOCKS
from urllib.parse import quote
import re 
import logging
# Set Google Gemini API Key
genai.configure(api_key="YOUR_OWN_KEY")

# Set page title
st.set_page_config(page_title="AI-Powered Stock Analysis", layout="wide")
st.title("📈 AI-Powered Stock Analysis and Prediction Tool")

# Load Data
@st.cache_data
def load_data():
    indian_df = pd.read_csv("indian_stocks_historical_data.csv")
    us_df = pd.read_csv("us_stocks_historical_data.csv")
    indian_df['Date'] = pd.to_datetime(indian_df['Date'])
    us_df['Date'] = pd.to_datetime(us_df['Date'])
    # Remove .NS from Indian stock tickers
    indian_df['Ticker'] = indian_df['Ticker'].str.replace('.NS', '', regex=False)
    return indian_df, us_df

indian_df, us_df = load_data()

# Create ticker lists
indian_tickers = sorted(indian_df['Ticker'].unique())
us_tickers = sorted(us_df['Ticker'].unique())

# Simplified navigation
selection = st.sidebar.radio("Go to:", [
    "Home",
    "Stock Prediction",
    "Quarterly Results Analysis",
    "Financial News",
    "Market Cap Suggestions",
    "Chatbot",
    "Equity Trend Analyzer",
    "Portfolio Optimizer",
    "Best Buy/Sell Timing"
])

# --- Home: Show Market Indices ---
if selection == "Home":
    st.header("🌍 Global Stock Market Indices")

    indices = {
        "US Markets": {
            "Dow Jones (DJIA)": "^DJI",
            "S&P 500": "^GSPC",
            "Nasdaq Composite": "^IXIC",
            "Russell 2000": "^RUT",
            "VIX (Volatility Index)": "^VIX"
        },
        "Indian Markets": {
            "Sensex": "^BSESN",
            "Nifty 50": "^NSEI",
            "Nifty Bank": "^NSEBANK",
            "Nifty Midcap 100": "^NSEMDCP50",
            "India VIX": "^INDIAVIX"
        }
    }

    col1, col2 = st.columns(2)

    def fetch_index_data(ticker):
        data = yf.Ticker(ticker).history(period="1d")
        return round(data['Close'].iloc[-1], 2) if not data.empty else "N/A"

    with col1:
        st.subheader("US Market Indices")
        for name, ticker in indices["US Markets"].items():
            price = fetch_index_data(ticker)
            st.metric(label=name, value=price)

    with col2:
        st.subheader("Indian Market Indices")
        for name, ticker in indices["Indian Markets"].items():
            price = fetch_index_data(ticker)
            st.metric(label=name, value=price)


# --- Feature: Stock Prediction with News Sentiment ---
elif selection == "Stock Prediction":
    st.header("📊 Stock Prediction")

    # Main container for inputs
    with st.container():
        st.subheader("Stock Selection")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            market_choice = st.radio("Select Market:", ["Indian Stock Market", "US Stock Market"])
        
        with col2:
            stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL for US, RELIANCE for India):")
            
    # Time frames mapping for buttons
    time_frames = {
        "1 Month": "1mo",
        "6 Months": "6mo",
        "1 Year": "1y",
        "2 Years": "2y", 
        "3 Years": "3y",
        "4 Years": "4y",
        "5 Years": "5y"
    }
    
    # Default selected time frame
    if 'selected_time_frame' not in st.session_state:
        st.session_state.selected_time_frame = "1 Month"

    # Data fetching functions
    @st.cache_data
    def fetch_stock_data(symbol, period):
        suffix = ".NS" if market_choice == "Indian Stock Market" else ""
        stock = yf.Ticker(f"{symbol}{suffix}")
        df = stock.history(period=period)
        df.reset_index(inplace=True)
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df.rename(columns={'Date': 'date', 'Open': 'open_price', 'High': 'high', 'Low': 'low', 
                          'Close': 'close_price', 'Volume': 'volume'}, inplace=True)
        return df

    def fetch_news_sentiment(symbol):
        url = f'https://newsapi.org/v2/everything?q={symbol}&apiKey=YOUR_NewsAPI_Key'
        response = requests.get(url)
        sentiment_score = 0
        count = 0
        if response.status_code == 200:
            articles = response.json().get('articles', [])[:5]
            for article in articles:
                analysis = TextBlob(article['title'])
                sentiment_score += analysis.sentiment.polarity
                count += 1
        return round(sentiment_score / count, 2) if count > 0 else "No News"

    def calculate_technical_indicators(df):
        if len(df) < 1:
            return df
        
        if len(df) >= 14:
            df['rsi'] = ta.momentum.RSIIndicator(df['close_price']).rsi()
        else:
            df['rsi'] = np.nan

        if len(df) >= 20:
            bollinger = ta.volatility.BollingerBands(df['close_price'])
            df['bollinger_mavg'] = bollinger.bollinger_mavg()
            df['bollinger_hband'] = bollinger.bollinger_hband()
            df['bollinger_lband'] = bollinger.bollinger_lband()
        else:
            df['bollinger_mavg'] = np.nan
            df['bollinger_hband'] = np.nan
            df['bollinger_lband'] = np.nan

        if len(df) >= 20:
            df['sma_20'] = ta.trend.SMAIndicator(df['close_price'], window=20).sma_indicator()
        else:
            df['sma_20'] = np.nan
        if len(df) >= 50:
            df['sma_50'] = ta.trend.SMAIndicator(df['close_price'], window=50).sma_indicator()
        else:
            df['sma_50'] = np.nan
        if len(df) >= 200:
            df['sma_200'] = ta.trend.SMAIndicator(df['close_price'], window=200).sma_indicator()
        else:
            df['sma_200'] = np.nan

        if len(df) >= 14:
            df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], 
                                                      df['close_price']).average_true_range()
        else:
            df['atr'] = np.nan
        
        return df

    # Prediction button and results
    if st.button("Predict Stock Price") and stock_symbol:
        with st.spinner("Fetching and analyzing data..."):
            period = time_frames[st.session_state.selected_time_frame]
            df = fetch_stock_data(stock_symbol, period)

            if not df.empty:
                df = calculate_technical_indicators(df)
                df['date'] = pd.to_datetime(df['date'])
                df['date_ordinal'] = df['date'].map(lambda x: x.toordinal())
                df_model = df.dropna(subset=['date_ordinal', 'open_price', 'volume', 
                                          'close_price'])
                
                if len(df_model) < 10:
                    st.warning("Not enough data points for reliable prediction.")
                else:
                    X = df_model[['date_ordinal', 'open_price', 'volume']]
                    y = df_model['close_price']
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                                      random_state=42)
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

                    next_day = pd.to_datetime(df['date'].max()) + pd.Timedelta(days=1)
                    next_day_ordinal = next_day.toordinal()
                    predicted_price = model.predict([[next_day_ordinal, 
                                                  df['open_price'].iloc[-1], 
                                                  df['volume'].iloc[-1]]])[0]

                    currency = "₹" if market_choice == "Indian Stock Market" else "$"

                    st.subheader(f"Analysis for {stock_symbol}")
                    
                    chart_col, metrics_col = st.columns([3, 1])

                    with chart_col:
                        price_tab, tech_tab = st.tabs(["Price History", "Technical Analysis"])
                        
                        with price_tab:
                            fig, ax = plt.subplots(figsize=(10, 5))
                            ax.plot(df['date'], df['close_price'], label="Close Price", 
                                   color="#1f77b4", linewidth=2, marker='o', markersize=4, 
                                   markeredgecolor='black')
                            ax.set_xlabel("Date", fontsize=12, color='darkblue')
                            ax.set_ylabel(f"Price ({currency})", fontsize=12, color='darkblue')
                            ax.set_title(f"{stock_symbol} Price History", fontsize=14, 
                                        fontweight='bold', color='navy')
                            ax.grid(True, linestyle='--', alpha=0.7, color='lightgrey')
                            ax.legend(loc='best', fontsize=10, frameon=True, facecolor='white', 
                                     edgecolor='black')
                            plt.xticks(rotation=45, fontsize=10)
                            plt.yticks(fontsize=10)
                            fig.patch.set_facecolor('#f0f0f0')
                            ax.set_facecolor('#fafafa')
                            st.pyplot(fig)
                            
                            # Time frame buttons
                            st.write("**Time Frame:**")
                            time_frame_cols = st.columns(len(time_frames))
                            for i, (time_frame, _) in enumerate(time_frames.items()):
                                with time_frame_cols[i]:
                                    is_active = st.session_state.selected_time_frame == time_frame
                                    button_type = "primary" if is_active else "secondary"
                                    if st.button(f"{time_frame}", key=f"btn_{time_frame}", type=button_type, use_container_width=True):
                                        st.session_state.selected_time_frame = time_frame
                                        st.rerun()

                        with tech_tab:
                            # Always use 2 plots (Price with SMA and RSI)
                            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
                            
                            ax1.plot(df['date'], df['close_price'], label="Close", 
                                    color="#ff7f0e", linewidth=2, alpha=0.9)
                            if 'sma_50' in df.columns and not df['sma_50'].isna().all():
                                ax1.plot(df['date'], df['sma_50'], label="SMA 50", 
                                        color="#2ca02c", linewidth=2, linestyle='--', alpha=0.8)
                            ax1.fill_between(df['date'], df['close_price'], df['sma_50'] if 'sma_50' in df.columns else df['close_price'], 
                                            alpha=0.1, color='orange')
                            ax1.set_ylabel(f"Price ({currency})", fontsize=12, color='darkgreen')
                            ax1.set_title("Price with Moving Average", fontsize=13, 
                                         fontweight='bold', color='darkgreen')
                            ax1.grid(True, linestyle='--', alpha=0.7, color='lightgrey')
                            ax1.legend(loc='best', fontsize=10, frameon=True, facecolor='white')

                            if 'rsi' in df.columns and not df['rsi'].isna().all():
                                ax2.plot(df['date'], df['rsi'], color="#d62728", linewidth=2, 
                                        marker='o', markersize=3, alpha=0.9)
                                ax2.axhline(y=70, color='#ff0000', linestyle='--', alpha=0.5, 
                                           label='Overbought (70)')
                                ax2.axhline(y=30, color='#00cc00', linestyle='--', alpha=0.5, 
                                           label='Oversold (30)')
                                ax2.fill_between(df['date'], 70, df['rsi'], where=(df['rsi'] >= 70), 
                                                color='red', alpha=0.2)
                                ax2.fill_between(df['date'], 30, df['rsi'], where=(df['rsi'] <= 30), 
                                                color='green', alpha=0.2)
                            ax2.set_ylabel("RSI", fontsize=12, color='darkred')
                            ax2.set_title("RSI", fontsize=13, fontweight='bold', color='darkred')
                            ax2.grid(True, linestyle='--', alpha=0.7, color='lightgrey')
                            ax2.legend(loc='best', fontsize=10, frameon=True, facecolor='white')

                            plt.xticks(rotation=45, fontsize=10)
                            plt.tight_layout()
                            fig.patch.set_facecolor('#f0f0f0')
                            for ax in (ax1, ax2):
                                ax.set_facecolor('#fafafa')
                            st.pyplot(fig)
                            
                            # Time frame buttons in technical tab too
                            st.write("**Time Frame:**")
                            tech_time_frame_cols = st.columns(len(time_frames))
                            for i, (time_frame, _) in enumerate(time_frames.items()):
                                with tech_time_frame_cols[i]:
                                    is_active = st.session_state.selected_time_frame == time_frame
                                    button_type = "primary" if is_active else "secondary"
                                    if st.button(f"{time_frame}", key=f"tech_btn_{time_frame}", type=button_type, use_container_width=True):
                                        st.session_state.selected_time_frame = time_frame
                                        st.rerun()

                    with metrics_col:
                        st.metric("Predicted Next Day Price", f"{currency}{round(predicted_price, 2)}")
                        st.metric("Model Accuracy (RMSE)", f"{round(rmse, 2)}")
                        
                        sentiment = fetch_news_sentiment(stock_symbol)
                        st.metric("News Sentiment", sentiment, 
                                 delta_color="normal" if sentiment == "No News" else 
                                 "inverse" if sentiment < 0 else "normal")

                        with st.expander("Technical Indicators", expanded=True):
                            latest = df.iloc[-1]
                            rsi = round(latest['rsi'], 2) if not np.isnan(latest['rsi']) else "N/A"
                            sma_20 = round(latest['sma_20'], 2) if not np.isnan(latest['sma_20']) else "N/A"
                            sma_50 = round(latest['sma_50'], 2) if not np.isnan(latest['sma_50']) else "N/A"
                            atr = round(latest['atr'], 2) if not np.isnan(latest['atr']) else "N/A"

                            st.markdown("**RSI (14)**")
                            if rsi != "N/A":
                                color = "red" if rsi > 70 else "green" if rsi < 30 else "grey"
                                st.markdown(f"<span style='color:{color}'>{rsi}</span>", unsafe_allow_html=True)
                            else:
                                st.write("N/A")

                            st.markdown("**SMA 20**")
                            st.write(f"{currency}{sma_20}")

                            st.markdown("**SMA 50**")
                            st.write(f"{currency}{sma_50}")

                            st.markdown("**ATR**")
                            st.write(f"{currency}{atr}")

                    with st.container():
                        st.subheader("Detailed Trend Analysis")
                        latest = df.iloc[-1]
                        analysis_points = []

                        if not np.isnan(latest['rsi']):
                            rsi_value = round(latest['rsi'], 2)
                            if rsi_value > 70:
                                analysis_points.append(f"**RSI ({rsi_value})**: Indicates potential **overbought** conditions, suggesting the stock might be due for a pullback.")
                            elif rsi_value < 30:
                                analysis_points.append(f"**RSI ({rsi_value})**: Indicates potential **oversold** conditions, which could signal a buying opportunity.")
                            else:
                                analysis_points.append(f"**RSI ({rsi_value})**: Currently in neutral territory, showing no extreme momentum.")

                        if not np.isnan(latest['sma_20']) and not np.isnan(latest['sma_50']):
                            sma_20_value = round(latest['sma_20'], 2)
                            sma_50_value = round(latest['sma_50'], 2)
                            if sma_20_value > sma_50_value:
                                analysis_points.append(f"**SMA Crossover ({currency}{sma_20_value} > {currency}{sma_50_value})**: Short-term trend is above medium-term trend, indicating a bullish signal.")
                            else:
                                analysis_points.append(f"**SMA Crossover ({currency}{sma_20_value} < {currency}{sma_50_value})**: Short-term trend is below medium-term trend, suggesting a bearish signal.")

                        if not np.isnan(latest['atr']):
                            atr_value = round(latest['atr'], 2)
                            analysis_points.append(f"**ATR ({currency}{atr_value})**: Measures volatility. Higher values suggest larger price swings, while lower values indicate consolidation.")

                        if analysis_points:
                            for point in analysis_points:
                                st.markdown(f"- {point}", unsafe_allow_html=True)
                            st.markdown("**Summary**: Consider these signals in context with market conditions and your investment strategy.")
                        else:
                            st.write("Insufficient data for a detailed trend analysis.")
            else:
                st.error("No data available for this symbol.")
    else:
        # Display empty time frame buttons when no stock is selected yet
        st.write("**Select Time Frame:**")
        time_frame_cols = st.columns(len(time_frames))
        for i, (time_frame, _) in enumerate(time_frames.items()):
            with time_frame_cols[i]:
                is_active = st.session_state.selected_time_frame == time_frame
                button_type = "primary" if is_active else "secondary"
                if st.button(f"{time_frame}", key=f"initial_btn_{time_frame}", type=button_type, use_container_width=True):
                    st.session_state.selected_time_frame = time_frame

# --- Feature: Financial News (Company Specific) ---
# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

if selection == "Financial News":  # Use 'if' assuming part of a larger Streamlit app
    st.header("📰 Company-Specific Financial News")

    market_type = st.radio("Choose Market:", ["Indian Market News", "US Market News"])

    symbol_input = st.text_input(
        "Enter Stock Symbol:",
        help="For Indian stocks like TCS, just enter 'TCS'. For US stocks like Apple, use 'AAPL'."
    )

    if st.button("Get News") and symbol_input:
        with st.spinner("Fetching news articles..."):
            symbol_input = symbol_input.strip().upper()
            
            # Handle Indian/US stock suffix
            if market_type == "Indian Market News":
                if not symbol_input.endswith(".NS"):
                    symbol = f"{symbol_input}.NS"
                else:
                    symbol = symbol_input
            else:
                symbol = symbol_input

            # Validate symbol
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                company_name = info.get("longName", "")
                if not company_name:
                    raise ValueError("Invalid symbol")
                st.session_state["last_symbol"] = symbol
            except Exception:
                st.error(f"Invalid stock symbol: {symbol}. Try valid tickers like TCS or AAPL.")
                st.stop()

            # Prepare query and regex
            query = f"{symbol_input} OR \"{company_name}\""
            pattern = re.compile(rf'\b({re.escape(symbol_input)}|{re.escape(company_name)})\b', re.IGNORECASE)

            # News API parameters
            newsapi_params = {
                'q': query,
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 10,
                'searchIn': 'title',
                'apiKey': 'YOUR_NewsAPI_Key'
            }

            # Fetch News API articles
            @st.cache_data(ttl=3600)
            def fetch_news(params):
                try:
                    response = requests.get("https://newsapi.org/v2/everything", params=params)
                    response.raise_for_status()
                    return response.json().get('articles', [])
                except Exception as e:
                    st.error(f"NewsAPI Error: {e}")
                    return []

            articles = fetch_news(newsapi_params)
            filtered_articles = []
            for article in articles:
                title = article.get('title', '')
                if pattern.search(title):
                    # Ensure description is a string
                    description = article.get('description')
                    if description is None:
                        logger.debug(f"Article with title '{title}' has None description")
                        article['description'] = ""
                    filtered_articles.append(article)

            # Fallback to Google News if <5
            if len(filtered_articles) < 5:
                try:
                    query_encoded = f"({symbol_input}+OR+\"{company_name}\")+(financial+OR+earnings+OR+stock)"
                    rss_url = f"https://news.google.com/rss/search?q={query_encoded}&hl=en-US&gl=US&ceid=US:en"
                    feed = feedparser.parse(rss_url)
                    for entry in feed.entries[:10]:
                        title = entry.get("title", "")
                        if not pattern.search(title):
                            continue
                        pub_date = None
                        if entry.get("published_parsed"):
                            pub_date = datetime.datetime(*entry.published_parsed[:6]).isoformat()
                        description = entry.get("description", "")
                        if description is None:
                            logger.debug(f"Google News article with title '{title}' has None description")
                            description = ""
                        filtered_articles.append({
                            "title": title,
                            "source": {"name": entry.get("source", {}).get("title", "Google News")},
                            "url": entry.get("link", "#"),
                            "publishedAt": pub_date,
                            "description": description
                        })
                except Exception as e:
                    st.error(f"Google News Fallback Failed: {e}")

            # Deduplication
            seen = set()
            final_articles = []
            for article in sorted(filtered_articles, key=lambda x: x.get('publishedAt', ''), reverse=True):
                url = article.get("url", "").split('?')[0].split('#')[0]
                if url and url not in seen and article.get('title') and article.get('source', {}).get('name'):
                    seen.add(url)
                    final_articles.append(article)

            # Display results
            if final_articles:
                st.subheader(f"📢 Latest News for {company_name} ({symbol})")
                for i, article in enumerate(final_articles[:5], 1):
                    pub_time = article.get("publishedAt", "")
                    try:
                        dt = datetime.datetime.fromisoformat(pub_time.replace("Z", ""))
                        pub_time_str = dt.strftime("%d %b %Y %H:%M UTC")
                    except:
                        pub_time_str = "Recent"

                    with st.expander(f"{i}. {article['title']} ({pub_time_str})", expanded=(i == 1)):
                        cols = st.columns([3, 1])
                        with cols[0]:
                            st.markdown(f"**Source**: {article['source']['name']}")
                            desc = article.get("description", "")
                            if not isinstance(desc, str):
                                logger.debug(f"Article {i} description is not a string: {desc}")
                                desc = ""
                            desc = desc.replace('\n', ' ') if desc else "No summary available"
                            st.markdown(f"**Summary**: {desc[:400]}..." if desc else "*No summary available*")
                        with cols[1]:
                            st.markdown(f"[📖 Read Full Article]({article['url']})", unsafe_allow_html=True)
                            try:
                                price = ticker.history(period='1d')['Close'].iloc[-1]
                                currency = "₹" if ".NS" in symbol else "$"
                                st.metric("Current Price", f"{currency}{price:.2f}")
                            except Exception as e:
                                logger.debug(f"Failed to fetch price for {symbol}: {e}")
                                pass
            else:
                st.warning(
                    f"No relevant news found for {symbol}. Try:\n"
                    "- Using correct ticker symbols (e.g., TCS for Indian, AAPL for US)\n"
                    "- Checking spelling or market selection"
                )

        st.button("🔄 Refresh News", key="refresh_news")

                
# --- Feature: Quarterly Results Analysis ---
elif selection == "Quarterly Results Analysis":
    st.header("📑 Quarterly Results Analysis")

    market_choice = st.radio("Select Market:", ["Indian Stock Market", "US Stock Market"])
    stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL for US, RELIANCE for India):")

    if st.button("Analyze") and stock_symbol:
        financials_data = {}
        currency = "₹" if market_choice == "Indian Stock Market" else "$"

        if market_choice == "US Stock Market":
            api_key = "YOUR_FINANCIAL_MODELING_PREP_KEY"
            url = f"https://financialmodelingprep.com/api/v3/income-statement/{stock_symbol}?limit=4&apikey={api_key}"
            balance_sheet_url = f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/{stock_symbol}?limit=1&apikey={api_key}"
            cash_flow_url = f"https://financialmodelingprep.com/api/v3/cash-flow-statement/{stock_symbol}?limit=1&apikey={api_key}"

            response_income = requests.get(url)
            response_balance = requests.get(balance_sheet_url)
            response_cashflow = requests.get(cash_flow_url)

            if response_income.status_code == 200 and response_balance.status_code == 200 and response_cashflow.status_code == 200:
                income_data, balance_data, cashflow_data = response_income.json(), response_balance.json(), response_cashflow.json()

                if income_data and balance_data and cashflow_data:
                    latest_income, latest_balance, latest_cashflow = income_data[0], balance_data[0], cashflow_data[0]

                    # Revenue
                    revenue = latest_income.get("revenue", "N/A")
                    financials_data["Revenue"] = f"{currency}{revenue:,}" if revenue != "N/A" else "N/A"

                    # Net Income
                    net_income = latest_income.get("netIncome", "N/A")
                    financials_data["Net Income"] = f"{currency}{net_income:,}" if net_income != "N/A" else "N/A"

                    # EPS
                    eps = latest_income.get("eps", "N/A")
                    financials_data["EPS"] = f"{eps:.2f}" if isinstance(eps, (int, float)) else "N/A"

                    # Debt-to-Equity Ratio (Total Liabilities / Total Shareholders' Equity)
                    total_liabilities = latest_balance.get("totalLiabilities", None)
                    if total_liabilities is None:
                        total_assets = latest_balance.get("totalAssets", 0)
                        equity = latest_balance.get("totalStockholdersEquity", 0)
                        total_liabilities = total_assets - equity if total_assets and equity else None

                    equity = latest_balance.get("totalStockholdersEquity", None)
                    if equity is None:
                        total_assets = latest_balance.get("totalAssets", 0)
                        total_liabilities = latest_balance.get("totalLiabilities", 0)
                        equity = total_assets - total_liabilities if total_assets and total_liabilities else None

                    debt_to_equity = f"{total_liabilities / equity:.2f}" if total_liabilities is not None and equity is not None and equity != 0 else "N/A"
                    financials_data["Debt-to-Equity Ratio"] = debt_to_equity

                    # Profit Margin
                    profit_margin = f"{(net_income / revenue) * 100:.2f}%" if revenue != "N/A" and net_income != "N/A" and revenue != 0 else "N/A"
                    financials_data["Profit Margin"] = profit_margin

                    # Free Cash Flow
                    operating_cash_flow = latest_cashflow.get("operatingCashFlow", "N/A")
                    capital_expenditure = latest_cashflow.get("capitalExpenditure", "N/A")
                    free_cash_flow = (
                        operating_cash_flow - capital_expenditure
                        if isinstance(operating_cash_flow, (int, float)) and isinstance(capital_expenditure, (int, float))
                        else "N/A"
                    )
                    financials_data["Free Cash Flow"] = f"{currency}{free_cash_flow:,}" if free_cash_flow != "N/A" else "N/A"

                    # Gross Margin (Gross Profit / Revenue * 100)
                    gross_profit = latest_income.get("grossProfit", "N/A")
                    gross_margin = f"{(gross_profit / revenue) * 100:.2f}%" if revenue != "N/A" and gross_profit != "N/A" and revenue != 0 else "N/A"
                    financials_data["Gross Margin"] = gross_margin

                    # Operating Margin (Operating Income / Revenue * 100)
                    operating_income = latest_income.get("operatingIncome", "N/A")
                    operating_margin = f"{(operating_income / revenue) * 100:.2f}%" if revenue != "N/A" and operating_income != "N/A" and revenue != 0 else "N/A"
                    financials_data["Operating Margin"] = operating_margin

                    # Current Ratio (Total Current Assets / Total Current Liabilities)
                    current_assets = latest_balance.get("totalCurrentAssets", None)
                    current_liabilities = latest_balance.get("totalCurrentLiabilities", None)
                    current_ratio = f"{current_assets / current_liabilities:.2f}" if current_assets is not None and current_liabilities is not None and current_liabilities != 0 else "N/A"
                    financials_data["Current Ratio"] = current_ratio

                    # Return on Equity (Net Income / Total Stockholders' Equity * 100)
                    roe = f"{(net_income / equity) * 100:.2f}%" if net_income != "N/A" and equity is not None and equity != 0 else "N/A"
                    financials_data["Return on Equity"] = roe

                    # Revenue Growth (Percentage change from previous quarter)
                    revenue_growth = "N/A"
                    if len(income_data) > 1:
                        prev_revenue = income_data[1].get("revenue", None)
                        if revenue != "N/A" and prev_revenue is not None and prev_revenue != 0:
                            revenue_growth = f"{((revenue - prev_revenue) / prev_revenue) * 100:.2f}%"
                    financials_data["Revenue Growth"] = revenue_growth

                    st.subheader(f"Financial Data for {stock_symbol} (Latest Quarter)")
                    for key, value in financials_data.items():
                        st.metric(label=key, value=value)

                    analysis_prompt = f"Analyze the latest quarterly financial results for {stock_symbol}: {financials_data}. Summarize the positives and negatives in exactly 10 lines, focusing on strengths and weaknesses based on the data. Do not provide investment suggestions, advice, or statements suggesting further analysis, investigation, or additional context. Provide clear, factual observations about the financial metrics in clean, professional text without formatting errors."
                    model = genai.GenerativeModel("gemini-1.5-flash-latest")
                    response = model.generate_content(analysis_prompt)

                    st.subheader("📢 Financial Analysis")
                    st.markdown(
                        f"""
                        <div style='font-family: Times New Roman; font-size: 16px;'>
                        {response.text}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.error("No financial data available. Please check the stock symbol.")
            else:
                st.error("Error fetching data. Try again later.")

        elif market_choice == "Indian Stock Market":
            try:
                stock = yf.Ticker(f"{stock_symbol}.NS")
                financials, balance_sheet, cashflow = stock.financials, stock.balance_sheet, stock.cashflow

                if not financials.empty or not balance_sheet.empty or not cashflow.empty:
                    # Revenue
                    revenue = financials.loc["Total Revenue"][0] if "Total Revenue" in financials.index else "N/A"
                    financials_data["Revenue"] = f"{currency}{revenue:,}" if revenue != "N/A" else "N/A"

                    # Net Income
                    net_income = financials.loc["Net Income"][0] if "Net Income" in financials.index else "N/A"
                    financials_data["Net Income"] = f"{currency}{net_income:,}" if net_income != "N/A" else "N/A"

                    # EPS
                    eps = financials.loc["Diluted EPS"][0] if "Diluted EPS" in financials.index else "N/A"
                    financials_data["EPS"] = f"{eps:.2f}" if isinstance(eps, (int, float)) else "N/A"

                    # Operating Margin (Operating Income / Total Revenue * 100)
                    operating_income = financials.loc["Operating Income"][0] if "Operating Income" in financials.index else "N/A"
                    operating_margin = f"{(operating_income / revenue) * 100:.2f}%" if revenue != "N/A" and operating_income != "N/A" and revenue != 0 else "N/A"
                    financials_data["Operating Margin"] = operating_margin

                    # Gross Margin (Gross Profit / Total Revenue * 100)
                    gross_profit = financials.loc["Gross Profit"][0] if "Gross Profit" in financials.index else "N/A"
                    gross_margin = f"{(gross_profit / revenue) * 100:.2f}%" if revenue != "N/A" and gross_profit != "N/A" and revenue != 0 else "N/A"
                    financials_data["Gross Margin"] = gross_margin

                    # Profit Margin
                    profit_margin = f"{(net_income / revenue) * 100:.2f}%" if revenue != "N/A" and net_income != "N/A" and revenue != 0 else "N/A"
                    financials_data["Profit Margin"] = profit_margin

                    st.subheader(f"Financial Data for {stock_symbol} (Latest Quarter)")
                    for key, value in financials_data.items():
                        st.metric(label=key, value=value)

                    analysis_prompt = f"Analyze the latest quarterly financial results for {stock_symbol}: {financials_data}. Summarize the positives and negatives in exactly 10 lines, focusing on strengths and weaknesses based on the data. Do not provide investment suggestions, advice, or statements suggesting further analysis, investigation, or additional context. Provide clear, factual observations about the financial metrics in clean, professional text without formatting errors."
                    model = genai.GenerativeModel("gemini-1.5-flash-latest")
                    response = model.generate_content(analysis_prompt)

                    st.subheader("📢 Financial Analysis")
                    st.markdown(
                        f"""
                        <div style='font-family: Times New Roman; font-size: 16px;'>
                        {response.text}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.error("No financial data available. Please check the stock symbol.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")


# --- Feature: Financial News (Company Specific) ---
elif selection == "Financial News":
    st.header("📰 Latest Company News")

    # Input field
    company_input = st.text_input("Enter Company Name or Stock Symbol (e.g., TCS, Tata, Apple, AAPL):")

    if st.button("Get News") and company_input:
        with st.spinner("Fetching the latest news..."):
            # Map common company names to full names for better search results
            company_mapping = {
                "tcs": "Tata Consultancy Services",
                "tata": "Tata Group",
                "apple": "Apple Inc.",
                "reliance": "Reliance Industries",
                # Add more mappings as needed
            }
            query = company_mapping.get(company_input.lower(), company_input)

            # Determine if input is likely a stock ticker (e.g., AAPL, RELIANCE.NS)
            is_ticker = any(company_input.upper() == t for t in ["AAPL", "RELIANCE.NS", "TCS.NS"]) or company_input.endswith(".NS")

            # Step 1: Try News API
            news_api_url = (
                f"https://newsapi.org/v2/everything?"
                f"q={query}&"
                f"language=en&"           # Restrict to English
                f"sortBy=publishedAt&"    # Sort by most recent
                f"apiKey=YOUR_NewsAPI_Key"
            )
            response = requests.get(news_api_url)
            articles = []

            if response.status_code == 200:
                articles = response.json().get('articles', [])[:5]  # Top 5 from News API

            # Step 2: Fallback to Google News RSS if News API yields insufficient results
            if len(articles) < 3:  # Arbitrary threshold to trigger fallback
                import feedparser
                google_news_url = f"https://news.google.com/rss/search?q={query}+company+news+-inurl:(stock+price+OR+stocks)&hl=en-US&gl=US&ceid=US:en"
                feed = feedparser.parse(google_news_url)
                google_articles = feed.entries[:5]  # Top 5 from Google News

                # Convert Google News entries to News API-like format
                for entry in google_articles:
                    articles.append({
                        "title": entry.get("title", "No title"),
                        "source": {"name": entry.get("source", {}).get("title", "Google News")},
                        "url": entry.get("link", "#"),
                        "publishedAt": entry.get("published", "N/A"),
                        "description": entry.get("summary", "No description available.")
                    })

            # Deduplicate articles by URL
            seen_urls = set()
            unique_articles = []
            for article in articles:
                url = article.get("url", "#")
                if url not in seen_urls:
                    seen_urls.add(url)
                    unique_articles.append(article)
            unique_articles = unique_articles[:5]  # Limit to 5 unique articles

            # Display results
            if unique_articles:
                st.subheader(f"Top 5 Latest News Articles for {company_input}")
                for i, article in enumerate(unique_articles, 1):
                    with st.expander(f"{i}. {article['title']} ({article['publishedAt'][:10] if article['publishedAt'] != 'N/A' else 'Date N/A'})"):
                        st.write(f"**Source**: {article['source']['name']}")
                        st.markdown(f"[Read more]({article['url']})", unsafe_allow_html=True)
                        description = article.get('description', 'No description available.')
                        if description is None:
                            description = "No description available."
                        summary = f"**Summary**: {description[:200]}..." if len(description) > 200 else f"**Summary**: {description}"
                        st.write(summary)
            else:
                st.warning(f"No recent news found for {company_input}. Try a different name or symbol.")

            # Handle News API errors
            if response.status_code != 200:
                st.error(f"News API error (Status Code: {response.status_code}). Showing Google News results if available.")
                
# Financial News Feature
elif selection == "Financial News":
    st.header("📰 Latest Company News")

    def is_financial_relevant(article, query, stock_name):
        """Check if article is relevant with stock name and symbol in title."""
        financial_keywords = ['earnings', 'stock', 'financial', 'market', 'revenue', 'profit', 'shares', 'dividend', 'acquisition', 'merger', 'investment', 'tariffs', 'growth', 'analyst']
        exclude_keywords = ['product', 'launch', 'device', 'consumer', 'retail', 'sale', 'gadget']
        title = article.get('title', '').lower()
        description = article.get('description', '').lower() or ''
        content = title + ' ' + description
        # Check if both query (symbol) and stock_name are in title
        query_in_title = query.lower() in title
        stock_name_in_title = stock_name.lower() in title
        # Require at least 1 financial keyword
        financial_score = sum(1 for kw in financial_keywords if kw in content)
        # Exclude any non-financial keywords
        exclude_score = sum(1 for kw in exclude_keywords if kw in content)
        # Ensure query or stock_name in content for context
        is_query_mentioned = query.lower() in content or stock_name.lower() in content
        return query_in_title and stock_name_in_title and financial_score >= 1 and exclude_score == 0 and is_query_mentioned

    def get_relevance_score(article):
        """Assign a relevance score based on financial keywords and source."""
        financial_sources = ['moneycontrol', 'reuters', 'bloomberg', 'financial times', 'economic times']
        title = article.get('title', '').lower()
        description = article.get('description', '').lower() or ''
        content = title + ' ' + description
        score = sum(2 for kw in ['earnings', 'stock', 'financial', 'market', 'revenue', 'tariffs'] if kw in content)
        score += sum(1 for kw in ['profit', 'shares', 'dividend', 'acquisition', 'merger', 'investment', 'growth', 'analyst'] if kw in content)
        source = article.get('source', {}).get('name', '').lower()
        if any(fs in source for fs in financial_sources):
            score += 3
        return score

    # Input fields
    company_input = st.text_input("Enter Stock Symbol (e.g., AMZN, ITC, AAPL):")
    stock_name_input = st.text_input("Enter Stock Name (e.g., Amazon, ITC Limited):")

    if st.button("Get News") and company_input and stock_name_input:
        with st.spinner("Fetching the latest news..."):
            query = company_input
            stock_name = stock_name_input

            # Generic exclusions for all companies
            exclude_terms = ' -product -launch -device -consumer -retail -sale -gadget'

            # Step 1: Try News API with broader query
            news_api_query = f"{query}+{stock_name}+(earnings OR stock OR financial OR revenue OR profit OR acquisition OR merger OR investment OR tariffs OR growth OR analyst){exclude_terms}"
            news_api_url = (
                f"https://newsapi.org/v2/everything?"
                f"q={quote(news_api_query)}&"
                f"language=en&"
                f"sortBy=publishedAt&"
                f"pageSize=100&"
                f"apiKey=YOUR_NewsAPI_Key"
            )
            response = requests.get(news_api_url)
            articles = []

            if response.status_code == 200:
                raw_articles = response.json().get('articles', [])
                articles = [a for a in raw_articles if is_financial_relevant(a, query, stock_name) and get_relevance_score(a) >= 2]

            # Step 2: Fallback to Google News RSS with domain restriction
            if len(articles) < 5:
                google_news_query = f"{query}+{stock_name}+(earnings OR stock OR financial OR revenue OR profit OR acquisition OR merger OR investment OR tariffs OR growth OR analyst){exclude_terms} site:news.moneycontrol.com OR site:reuters.com OR site:bloomberg.com OR site:ft.com OR site:economictimes.indiatimes.com"
                google_news_url = (
                    f"https://news.google.com/rss/search?"
                    f"q={quote(google_news_query)}&"
                    f"hl=en-US&gl=US&ceid=US:en"
                )
                feed = feedparser.parse(google_news_url)
                google_articles = feed.entries[:50]  # Fetch more to filter

                for entry in google_articles:
                    article = {
                        'title': entry.get('title', 'No title'),
                        'source': {'name': entry.get('source', {}).get('title', 'Google News')},
                        'url': entry.get('link', '#'),
                        'publishedAt': entry.get('published', 'N/A'),
                        'description': entry.get('summary', 'No description available.')
                    }
                    if is_financial_relevant(article, query, stock_name) and get_relevance_score(article) >= 2:
                        articles.append(article)

            # Filter for top-tier financial sources
            financial_sources = ['moneycontrol', 'reuters', 'bloomberg', 'financial times', 'economic times']
            articles = [a for a in articles if any(fs in a.get('source', {}).get('name', '').lower() for fs in financial_sources)]

            # Deduplicate and sort by relevance
            seen_urls = set()
            unique_articles = []
            for article in articles:
                url = article.get('url', '#')
                if url not in seen_urls:
                    seen_urls.add(url)
                    unique_articles.append(article)
            unique_articles = sorted(unique_articles, key=get_relevance_score, reverse=True)

            # If fewer than 5 articles, relax score and retry
            if len(unique_articles) < 5:
                articles = []
                if response.status_code == 200:
                    articles += [a for a in raw_articles if is_financial_relevant(a, query, stock_name) and get_relevance_score(a) >= 1]
                articles += [a for a in google_articles if is_financial_relevant({
                    'title': a.get('title', 'No title'),
                    'source': {'name': a.get('source', {}).get('title', 'Google News')},
                    'url': a.get('link', '#'),
                    'publishedAt': a.get('published', 'N/A'),
                    'description': a.get('summary', 'No description available.')
                }, query, stock_name) and get_relevance_score(a) >= 1]
                unique_articles = []
                seen_urls = set()
                for article in articles:
                    url = article.get('url', '#')
                    if url not in seen_urls:
                        seen_urls.add(url)
                        unique_articles.append(article)
                unique_articles = sorted(unique_articles, key=get_relevance_score, reverse=True)[:5]

            # Display results
            st.subheader(f"Latest Financial News Articles for {company_input}")
            for i, article in enumerate(unique_articles[:5], 1):
                date_str = article['publishedAt'][:10] if article['publishedAt'] != 'N/A' else 'Date N/A'
                with st.expander(f"{i}. {article['title']} ({date_str})"):
                    st.write(f"**Source**: {article['source']['name']}")
                    st.markdown(f"[Read more]({article['url']})", unsafe_allow_html=True)
                    description = article.get('description', 'No description available.') or 'No description available.'
                    summary = f"**Summary**: {description[:200]}..." if len(description) > 200 else f"**Summary**: {description}"
                    st.write(summary)

            # Handle News API errors
            if response.status_code != 200:
                st.error(f"News API error (Status Code: {response.status_code}). Showing Google News results if available.")
                
#Feature---> Market Cap Suggestions             
elif selection == "Market Cap Suggestions":
    st.header("💰 Market Cap Suggestions")

    from functools import lru_cache
    import time
    from concurrent.futures import ThreadPoolExecutor
    from fallback_stocks import FALLBACK_STOCKS
    import json
    from google.generativeai.types import HarmCategory, HarmBlockThreshold

    # Set your Gemini API key
    GEMINI_API_KEY = "YOUR_GEMINI_KEY"

    # Initialize Gemini API
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }
        model = genai.GenerativeModel('gemini-pro', safety_settings=safety_settings)
        gemini_available = True
    except Exception as e:
        st.sidebar.error(f"Gemini API initialization failed: {str(e)}")
        gemini_available = False

    # Helper function to run async code in a separate thread
    def run_async_in_thread(coro):
        def run_coroutine():
            return asyncio.run(coro)
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_coroutine)
            return future.result()

    # Load and validate CSV files
    try:
        nifty500 = pd.read_csv("nifty500_companies.csv")
        us_historical = pd.read_csv("us_stocks_historical_data.csv")
        required_nifty_cols = ['Symbol', 'Company Name', 'Industry']
        required_us_cols = ['Ticker', 'Date', 'Close']
        if not all(col in nifty500.columns for col in required_nifty_cols):
            st.error(f"nifty500_companies.csv is missing required columns: {required_nifty_cols}")
            st.stop()
        if not all(col in us_historical.columns for col in required_us_cols):
            st.error(f"us_stocks_historical_data.csv is missing required columns: {required_us_cols}")
            st.stop()
        if nifty500.empty or us_historical.empty:
            st.error("One or both CSV files are empty.")
            st.stop()
    except Exception as e:
        st.error(f"Error loading CSV files: {e}")
        st.stop()

    # Initialize session state for tracking used symbols
    if 'used_symbols' not in st.session_state:
        st.session_state.used_symbols = set()

    # Define market cap ranges
    CAP_RANGES = {
        "Large Cap": {"US": (10, float('inf')), "India": (20000, float('inf'))},
        "Mid Cap": {"US": (2, 10), "India": (5000, 20000)},
        "Small Cap": {"US": (0.3, 2), "India": (500, 5000)}
    }

    # Define sectors
    SECTORS = {
        "Indian Stock Market": ["Technology", "Finance", "Energy", "Healthcare", "Consumer Goods", "Automobile"],
        "US Stock Market": ["Technology", "Finance", "Energy", "Healthcare", "Consumer Goods", "Industrials"]
    }

    # Async function to fetch Yahoo Finance data with timeout
    async def fetch_yf_data(symbol, timeout=5):
        try:
            async with aiohttp.ClientSession() as session:
                ticker = yf.Ticker(symbol)
                info = await asyncio.wait_for(
                    asyncio.to_thread(lambda: ticker.info), timeout=timeout
                )
                return info
        except Exception:
            return None

    # Async function to fetch historical data
    async def fetch_historical_data(symbol, period="1y"):
        try:
            ticker = yf.Ticker(symbol)
            hist = await asyncio.to_thread(lambda: ticker.history(period=period))
            return hist
        except Exception:
            return pd.DataFrame()

    # Generate enhanced stock description using Gemini API
    def get_gemini_analysis(stock_data, market, cap, sector, risk, goal):
        if not gemini_available:
            return None
        
        try:
            prompt = f"""
            Generate a detailed but concise analysis of this stock for an investor.
            
            Stock Information:
            - Name: {stock_data['name']}
            - Symbol: {stock_data['symbol']}
            - Market: {market}
            - Market Cap: {stock_data['market_cap']} {"crores (₹)" if market == "Indian Stock Market" else "billion ($)"}
            - Current Price: {stock_data['price']}
            - P/E Ratio: {stock_data['pe_ratio']}
            - Dividend Yield: {stock_data['dividend_yield']}%
            - 1-Year Performance: {stock_data['yearly_change']}%
            - 3-Month Performance: {stock_data['short_term_change']}%
            - Volatility: {stock_data['volatility']}%
            - Beta: {stock_data['beta']}
            - Industry: {stock_data['industry']}
            - Description: {stock_data['description']}
            
            Investor Preferences:
            - Market Cap: {cap}
            - Sector: {sector}
            - Risk Tolerance: {risk}
            - Investment Goal: {goal}
            
            Format your response in these sections:
            1. Company Details: Brief summary of what the company does, its market position, competitive advantages, and recent developments.
            2. Financial Analysis: Analysis of the company's financial health, growth prospects, valuation metrics, and revenue streams.
            3. Why Suggested: Explain why this stock matches the investor's preferences (market cap, sector, risk tolerance, and investment goal). Include any specific advantages or considerations.
            4. Investment Considerations: List key factors the investor should consider before investing (risks, potential catalysts, ideal holding period).
            
            Keep your analysis factual, balanced, and helpful for investment decisions. Total response should be 3-4 short paragraphs.
            """
            
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Gemini API error: {e}")
            return None

    # Process stock data asynchronously
    @lru_cache(maxsize=1000)
    async def process_stock(symbol, market, min_cap, max_cap, sector, risk, goal):
        try:
            market_type = "US" if market == "US Stock Market" else "India"
            divisor = 10000000 if market == "Indian Stock Market" else 1000000000
            currency = "₹" if market == "Indian Stock Market" else "$"
            yf_symbol = symbol + ".NS" if market == "Indian Stock Market" else symbol

            if yf_symbol in st.session_state.used_symbols:
                return None

            if market == "Indian Stock Market":
                hist_data = nifty500[nifty500['Symbol'] == symbol]
                if hist_data.empty:
                    return None
                name = hist_data['Company Name'].iloc[0]
                industry = hist_data['Industry'].iloc[0]
                info = await fetch_yf_data(yf_symbol)
                if not info or 'marketCap' not in info:
                    return None
                market_cap = info.get('marketCap', 0) / divisor
                price = info.get('regularMarketPrice', 0)
                pe_ratio = info.get('trailingPE', 'N/A')
                dividend_yield = info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
                beta = info.get('beta', 'N/A')
                week52_high = info.get('fiftyTwoWeekHigh', 'N/A')
                week52_low = info.get('fiftyTwoWeekLow', 'N/A')
                avg_volume = info.get('averageVolume', 'N/A')
                description = info.get('longBusinessSummary', 'No description available.')
                hist = await fetch_historical_data(yf_symbol)
                if hist.empty:
                    return None
                yearly_change = ((hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100 if len(hist) > 20 else 0
                volatility = hist['Close'].pct_change().std() * np.sqrt(252) * 100
                short_term_change = ((hist['Close'].iloc[-1] - hist['Close'].iloc[-90]) / hist['Close'].iloc[-90]) * 100 if len(hist) >= 90 else yearly_change
            else:
                hist_data = us_historical[us_historical['Ticker'] == symbol]
                if hist_data.empty:
                    return None
                name = symbol
                info = await fetch_yf_data(symbol)
                if not info or 'marketCap' not in info:
                    return None
                name = info.get('longName', symbol)
                industry = info.get('industry', '')
                market_cap = info.get('marketCap', 0) / divisor
                price = info.get('regularMarketPrice', 0)
                pe_ratio = info.get('trailingPE', 'N/A')
                dividend_yield = info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
                beta = info.get('beta', 'N/A')
                week52_high = info.get('fiftyTwoWeekHigh', 'N/A')
                week52_low = info.get('fiftyTwoWeekLow', 'N/A')
                avg_volume = info.get('averageVolume', 'N/A')
                description = info.get('longBusinessSummary', 'No description available.')
                hist_data['Date'] = pd.to_datetime(hist_data['Date'])
                hist_data = hist_data.sort_values('Date')
                yearly_change = ((hist_data['Close'].iloc[-1] - hist_data['Close'].iloc[0]) / hist_data['Close'].iloc[0]) * 100 if len(hist_data) > 20 else 0
                volatility = hist_data['Close'].pct_change().std() * np.sqrt(252) * 100
                short_term_change = ((hist_data['Close'].iloc[-1] - hist_data['Close'].iloc[-90]) / hist_data['Close'].iloc[-90]) * 100 if len(hist_data) >= 90 else yearly_change

            if not (min_cap <= market_cap <= max_cap):
                return None

            sector_mapping = {
                "Technology": ["Technology", "Information Technology", "Software", "IT Services"],
                "Finance": ["Financial Services", "Banks", "Insurance", "Financials"],
                "Energy": ["Energy", "Oil & Gas", "Utilities", "Power"],
                "Healthcare": ["Healthcare", "Pharmaceuticals", "Biotechnology", "Medical"],
                "Consumer Goods": ["Consumer Defensive", "Consumer Cyclical", "Retail", "FMCG"],
                "Automobile": ["Automobile", "Auto", "Vehicle"],
                "Industrials": ["Industrials", "Manufacturing", "Aerospace", "Defense"]
            }
            sector_keywords = sector_mapping.get(sector, [sector])
            sector_match = any(keyword.lower() in industry.lower() for keyword in sector_keywords)
            if not sector_match:
                return None

            risk_match = (risk == "Low" and volatility < 30) or \
                         (risk == "Moderate" and 20 <= volatility <= 40) or \
                         (risk == "High" and volatility > 25)
            if not risk_match:
                return None

            if goal == "Long Term":
                goal_match = volatility < 35 and yearly_change > -10 and (pe_ratio == 'N/A' or pe_ratio < 40)
            elif goal == "Medium Term":
                goal_match = yearly_change > -5 and volatility < 40
            else:  # Short Term
                goal_match = short_term_change > 5
            if not goal_match:
                return None

            score = 0
            if risk_match:
                score += 1
            if goal_match:
                score += 1
            if goal == "Long Term" and volatility < 25 and pe_ratio != 'N/A' and pe_ratio < 25:
                score += 0.5
            if goal == "Medium Term" and yearly_change > 5:
                score += 0.5
            if goal == "Short Term" and short_term_change > 15:
                score += 0.5

            st.session_state.used_symbols.add(yf_symbol)

            return {
                "symbol": symbol,
                "name": name,
                "market_cap": market_cap,
                "price": price,
                "yearly_change": yearly_change,
                "volatility": volatility,
                "pe_ratio": pe_ratio,
                "dividend_yield": dividend_yield,
                "beta": beta,
                "week52_high": week52_high,
                "week52_low": week52_low,
                "avg_volume": avg_volume,
                "description": description,
                "industry": industry,
                "score": score,
                "short_term_change": short_term_change
            }
        except Exception:
            return None

    # Fetch and process stocks with timeout
    async def fetch_stocks(market, cap, sector, risk, goal):
        market_type = "US" if market == "US Stock Market" else "India"
        min_cap, max_cap = CAP_RANGES[cap][market_type]
        
        if market == "Indian Stock Market":
            stock_list = nifty500['Symbol'].tolist()
        else:
            stock_list = us_historical['Ticker'].unique().tolist()
        
        random.shuffle(stock_list)
        
        stocks = []
        processed = 0
        max_process = 100
        
        for symbol in stock_list:
            if processed >= max_process:
                break
            result = await process_stock(symbol, market, min_cap, max_cap, sector, risk, goal)
            if result:
                stocks.append(result)
            processed += 1
        
        if len(stocks) < 5:
            fallback = FALLBACK_STOCKS[market][cap][sector][(risk, goal)]
            random.shuffle(fallback)
            for symbol in fallback[:10 - len(stocks)]:
                if symbol not in [s['symbol'] for s in stocks] and (symbol + ".NS" if market == "Indian Stock Market" else symbol) not in st.session_state.used_symbols:
                    result = await process_stock(symbol, market, min_cap, max_cap, sector, risk, goal)
                    if result:
                        stocks.append(result)
        
        if goal == "Long Term":
            sorted_stocks = sorted(stocks, key=lambda x: (x["score"], -x["volatility"]), reverse=True)
        elif goal == "Medium Term":
            sorted_stocks = sorted(stocks, key=lambda x: (x["score"], x["yearly_change"]), reverse=True)
        else:  # Short Term
            sorted_stocks = sorted(stocks, key=lambda x: (x["score"], x["short_term_change"]), reverse=True)
        
        return sorted_stocks[:10]

    # Function to get company name from CSV
    def get_company_name(symbol, market):
        if market == "Indian Stock Market":
            hist_data = nifty500[nifty500['Symbol'] == symbol]
            return hist_data['Company Name'].iloc[0] if not hist_data.empty else symbol
        else:
            hist_data = us_historical[us_historical['Ticker'] == symbol]
            return symbol if hist_data.empty else symbol

    # Function to display fallback stocks
    def display_fallback_stocks(market, cap, sector, risk, goal):
        fallback_stocks = FALLBACK_STOCKS[market][cap][sector][(risk, goal)]
        random.shuffle(fallback_stocks)
        
        for i, symbol in enumerate(fallback_stocks[:10], 1):
            name = get_company_name(symbol, market)
            yf_symbol = symbol + ".NS" if market == "Indian Stock Market" else symbol
            
            if yf_symbol in st.session_state.used_symbols:
                continue
            
            st.session_state.used_symbols.add(yf_symbol)
            
            st.markdown(f"**{i}. {name} ({symbol})**")
            
            with st.expander("Company Details"):
                st.markdown("### Company Details")
                st.markdown(f"- **Name**: {name}")
                st.markdown(f"- **Symbol**: {symbol}")
                st.markdown(f"- **Sector**: {sector}")
                
                # Generate Gemini analysis if API key is available
                if gemini_available:
                    fallback_data = {
                        "symbol": symbol,
                        "name": name,
                        "market_cap": "N/A",
                        "price": "N/A",
                        "yearly_change": "N/A",
                        "volatility": "N/A",
                        "pe_ratio": "N/A",
                        "dividend_yield": "N/A",
                        "beta": "N/A",
                        "week52_high": "N/A",
                        "week52_low": "N/A",
                        "avg_volume": "N/A",
                        "description": f"Predefined stock for {cap} in {sector} sector",
                        "industry": sector
                    }
                    with st.spinner("Generating AI analysis..."):
                        analysis = get_gemini_analysis(fallback_data, market, cap, sector, risk, goal)
                        if analysis:
                            st.markdown("### AI-Powered Analysis")
                            st.markdown(analysis)

    # Input Section
    with st.container():
        st.subheader("Your Investment Preferences")
        col1, col2 = st.columns(2)
        
        with col1:
            market_choice = st.radio("Select Market:", ["Indian Stock Market", "US Stock Market"])
            cap_choice = st.radio("Select Market Cap:", ["Large Cap", "Mid Cap", "Small Cap"])
        
        with col2:
            sector = st.selectbox("Select Sector:", SECTORS[market_choice])
            risk_tolerance = st.selectbox("Risk Tolerance:", ["Low", "Moderate", "High"])
            investment_goal = st.selectbox("Investment Goal:", ["Long Term", "Medium Term", "Short Term"])

    if st.button("Get Suggestions"):
        with st.spinner("Fetching stock data, please wait..."):
            time.sleep(5)  # Display spinner for 5 seconds
            st.session_state.used_symbols.clear()
            
            currency = "₹" if market_choice == "Indian Stock Market" else "$"
            
            try:
                stocks = run_async_in_thread(
                    asyncio.wait_for(
                        fetch_stocks(market_choice, cap_choice, sector, risk_tolerance, investment_goal),
                        timeout=10
                    )
                )
            except (asyncio.TimeoutError, Exception):
                stocks = None  # Set stocks to None to trigger fallback display
        
        # Display stocks outside spinner context to ensure spinner stops
        if not stocks:
            display_fallback_stocks(market_choice, cap_choice, sector, risk_tolerance, investment_goal)
        else:
            st.subheader(f"Top Stock Suggestions - {cap_choice}, {sector} ({market_choice})")
            
            for i, stock in enumerate(stocks, 1):
                st.markdown(f"**{i}. {stock['name']} ({stock['symbol']})**")
                
                with st.expander("Company Details"):
                    # Create tabs for different sections
                    basic_tab, financial_tab, gemini_tab = st.tabs(["Basic Info", "Financial Data", "AI Analysis"])
                    
                    with basic_tab:
                        st.markdown("### Company Details")
                        st.markdown(f"- **Name**: {stock['name']}")
                        st.markdown(f"- **Symbol**: {stock['symbol']}")
                        st.markdown(f"- **Sector**: {sector}")
                        st.markdown(f"- **Industry**: {stock['industry']}")
                        st.markdown(f"- **Description**: {stock['description'][:500] + '...' if len(stock['description']) > 500 else stock['description']}")
                    
                    with financial_tab:
                        st.markdown("### Financial Details")
                        st.markdown(f"- **Market Cap**: {currency}{round(stock['market_cap'], 2)} {' crores' if market_choice == 'Indian Stock Market' else ' billion'}")
                        st.markdown(f"- **Current Price**: {currency}{round(stock['price'], 2)}")
                        st.markdown(f"- **1-Year Performance**: {round(stock['yearly_change'], 2)}%")
                        st.markdown(f"- **3-Month Performance**: {round(stock['short_term_change'], 2)}%")
                        st.markdown(f"- **Volatility**: {round(stock['volatility'], 2)}% (annualized)")
                        st.markdown(f"- **P/E Ratio**: {stock['pe_ratio'] if stock['pe_ratio'] != 'N/A' else 'N/A'}")
                        st.markdown(f"- **Dividend Yield**: {round(stock['dividend_yield'], 2)}% {'(N/A)' if stock['dividend_yield'] <= 0 else ''}")
                        st.markdown(f"- **Beta**: {stock['beta'] if stock['beta'] != 'N/A' else 'N/A'}")
                        st.markdown(f"- **52-Week High**: {currency}{stock['week52_high'] if stock['week52_high'] != 'N/A' else 'N/A'}")
                        st.markdown(f"- **52-Week Low**: {currency}{stock['week52_low'] if stock['week52_low'] != 'N/A' else 'N/A'}")
                        st.markdown(f"- **Average Volume**: {stock['avg_volume'] if stock['avg_volume'] != 'N/A' else 'N/A'} shares")
                        
                    with gemini_tab:
                        if gemini_available:
                            with st.spinner("Generating AI analysis..."):
                                analysis = get_gemini_analysis(stock, market_choice, cap_choice, sector, risk_tolerance, investment_goal)
                                if analysis:
                                    st.markdown("### AI-Powered Analysis")
                                    st.markdown(analysis)
                                else:
                                    st.markdown("### Why Suggested")
                                    # Generate justification text
                                    justification = ""
                                    if risk_tolerance == "Low" and stock['volatility'] < 30:
                                        justification += "Low volatility aligns with your risk tolerance. "
                                    elif risk_tolerance == "Moderate" and 20 <= stock['volatility'] <= 40:
                                        justification += "Moderate volatility matches your risk profile. "
                                    elif risk_tolerance == "High" and stock['volatility'] > 25:
                                        justification += "Higher volatility suits your risk appetite. "
                                    
                                    if investment_goal == "Long Term":
                                        justification += f"Stable long-term potential with low volatility ({round(stock['volatility'], 2)}%) and reasonable valuation (P/E: {stock['pe_ratio']}). "
                                    elif investment_goal == "Medium Term":
                                        justification += f"Balanced growth potential with {round(stock['yearly_change'], 2)}% return over the past year. "
                                    else:  # Short Term
                                        justification += f"Strong short-term performance with {round(stock['short_term_change'], 2)}% return over the past 3 months. "
                                    
                                    cap_text = f"{cap_choice} ({currency}{round(stock['market_cap'], 1)} {' crores' if market_choice == 'Indian Stock Market' else ' billion'})"
                                    justification += f"Matches your {cap_text} preference in the {sector} sector."
                                    st.markdown(justification)
                        else:
                            st.markdown("### Why Suggested")
                            # Generate justification text
                            justification = ""
                            if risk_tolerance == "Low" and stock['volatility'] < 30:
                                justification += "Low volatility aligns with your risk tolerance. "
                            elif risk_tolerance == "Moderate" and 20 <= stock['volatility'] <= 40:
                                justification += "Moderate volatility matches your risk profile. "
                            elif risk_tolerance == "High" and stock['volatility'] > 25:
                                justification += "Higher volatility suits your risk appetite. "
                            
                            if investment_goal == "Long Term":
                                justification += f"Stable long-term potential with low volatility ({round(stock['volatility'], 2)}%) and reasonable valuation (P/E: {stock['pe_ratio']}). "
                            elif investment_goal == "Medium Term":
                                justification += f"Balanced growth potential with {round(stock['yearly_change'], 2)}% return over the past year. "
                            else:  # Short Term
                                justification += f"Strong short-term performance with {round(stock['short_term_change'], 2)}% return over the past 3 months. "
                            
                            cap_text = f"{cap_choice} ({currency}{round(stock['market_cap'], 1)} {' crores' if market_choice == 'Indian Stock Market' else ' billion'})"
                            justification += f"Matches your {cap_text} preference in the {sector} sector."
                            st.markdown(justification)
                            st.info("Enter a valid Gemini API key in the code to enable AI-powered analysis.")
                            
# --- Feature: Chatbot ---
elif selection == "Chatbot":
    st.header("🤖 Finbot - Your Finance Assistant")
    
    user_query = st.text_input("Ask about stocks, finance, or investments:")

    if st.button("Get Response") and user_query:
        # Expanded finance-related keywords (covering virtually all finance terms)
        finance_keywords = [
            "stock", "stocks", "share", "shares", "equity", " equities", "market", "markets",
            "bull", "bear", "trend", "index", "indices", "exchange", "nse", "bse", "nasdaq",
            "dow", "s&p", "sensex", "nifty", "ipo", "listing", "trade", "trading", "trader",
            "broker", "brokerage", "portfolio", "dividend", "dividends", "yield", "payout",
            "earnings", "revenue", "profit", "loss", "income", "expense", "margin", "gross",
            "net", "eps", "earnings per share", "pe", "p/e", "price-to-earnings", "valuation",
            "cap", "capital", "market cap", "large cap", "mid cap", "small cap", "asset",
            "assets", "liability", "liabilities", "debt", "loan", "leverage", "equity",
            "balance sheet", "cash flow", "operating", "financing", "investing", "quarter",
            "quarterly", "annual", "report", "financial", "finance", "investment", "invest",
            "investor", "fund", "funds", "mutual fund", "etf", "exchange-traded", "bond",
            "bonds", "treasury", "yield curve", "interest", "rate", "rates", "inflation",
            "deflation", "economy", "economic", "gdp", "recession", "boom", "cycle",
            "option", "options", "call", "put", "future", "futures", "derivative",
            "commodity", "commodities", "gold", "oil", "currency", "forex", "exchange rate",
            "risk", "volatility", "beta", "alpha", "return", "returns", "roi", "sharpe",
            "hedge", "hedging", "short", "long", "position", "buy", "sell", "hold",
            "diversify", "diversification", "sector", "industry", "tech", "technology",
            "energy", "healthcare", "consumer", "auto", "automobile", "bank", "banking",
            "insurance", "real estate", "reit", "mortgage", "credit", "debit", "liquidity",
            "solvency", "growth", "value", "blue chip", "penny", "speculation", "arbitrage",
            "technical", "analysis", "chart", "pattern", "trendline", "support", "resistance",
            "rsi", "macd", "bollinger", "moving average", "sma", "ema", "fundamental",
            "intrinsic", "book value", "cash", "money", "wealth", "tax", "taxes", "deduction",
            "expense", "budget", "fiscal", "monetary", "policy", "fed", "rbi", "central bank",
            "quantitative", "easing", "stimulus", "crash", "bubble", "correction", "rally"
        ]
        
        # Function to check if query is finance-related
        def is_finance_related(query):
            from textblob import TextBlob
            
            # Convert query to lowercase
            query_lower = query.lower()
            blob = TextBlob(query_lower)
            words = set(blob.words)
            
            # Step 1: Check for direct keyword matches
            if any(keyword in query_lower for keyword in finance_keywords):
                return True
            
            # Step 2: Analyze context using simple NLP (nouns and sentiment)
            finance_nouns = {"stock", "market", "investment", "fund", "bond", "share", 
                           "portfolio", "dividend", "profit", "loss", "revenue", "equity",
                           "debt", "asset", "liability", "broker", "exchange", "economy"}
            
            # Check if any finance-related nouns are present
            if words & finance_nouns:  # Intersection of sets
                return True
            
            # Step 3: Broader context check (e.g., verbs like "invest", "trade")
            finance_verbs = {"invest", "trade", "buy", "sell", "hold", "diversify", 
                           "analyze", "predict", "grow", "hedge"}
            if any(word in finance_verbs for word in words):
                # Ensure it’s not a non-finance context (e.g., "buy a car")
                non_finance_indicators = {"weather", "food", "cook", "travel", "game", 
                                       "movie", "sport", "music", "health", "doctor", 
                                       "pet", "dog", "cat"}
                if not any(indicator in query_lower for indicator in non_finance_indicators):
                    return True
            
            # Step 4: Fallback - ask Gemini to classify (lightweight check)
            try:
                model = genai.GenerativeModel("gemini-1.5-flash-latest")
                prompt = f"Is this question related to finance, stocks, or investments? Answer only 'Yes' or 'No': {query}"
                response = model.generate_content(prompt)
                return response.text.strip() == "Yes"
            except Exception:
                return False  # Default to false if API fails
        
        # Check if the query is finance-related
        if is_finance_related(user_query):
            with st.spinner("Generating response..."):
                model = genai.GenerativeModel("gemini-1.5-flash-latest")
                # Add instruction to ensure finance-only responses
                full_prompt = f"You are a financial expert. Provide an answer related to stocks, finance, or investments only. Query: {user_query}"
                response = model.generate_content(full_prompt)
                
                st.subheader("Response:")
                st.write(response.text)
        else:
            st.subheader("Response:")
            st.write("I’m here to help with finance-related questions only! Please ask about stocks, markets, investments, or similar topics. For example: 'What’s a good ETF to buy?' or 'How does RSI work?'")

# --- Feature: Equity Trend Analyzer ---
elif selection == "Equity Trend Analyzer":
    st.header("📅 Equity Trend Analyzer")
    
    # Input Section
    with st.container():
        st.subheader("Investment Scenario")
        col1, col2 = st.columns(2)
        
        with col1:
            market_choice = st.radio("Select Market:", ["Indian Stock Market", "US Stock Market"])
            stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL for US, RELIANCE for India):")
        
        with col2:
            today = pd.to_datetime("2025-04-08")  # Current date as per your instructions
            start_date = st.date_input("Start Date", value=pd.to_datetime("2024-01-01"), max_value=today)
            end_date = st.date_input("End Date", value=today, max_value=today)
            investment_amount = st.number_input("Daily Investment Amount ($ or ₹)", min_value=1.0, value=100.0, step=10.0)

    # Function to fetch stock data
    @st.cache_data
    def fetch_stock_data_range(symbol, start, end):
        suffix = ".NS" if market_choice == "Indian Stock Market" else ""
        try:
            stock = yf.Ticker(f"{symbol}{suffix}")
            df = stock.history(start=start, end=end + pd.Timedelta(days=1))  # Include end date
            if df.empty:
                st.warning(f"No data returned for {symbol} between {start} and {end}.")
                return None
            df.reset_index(inplace=True)
            df = df[['Date', 'Close']]
            df.rename(columns={'Date': 'date', 'Close': 'close_price'}, inplace=True)
            df['date'] = pd.to_datetime(df['date']).dt.date
            return df
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")
            return None

    # Calculate performance
    def calculate_performance(df, start, end, amount):
        if df is None or df.empty:
            return None, None, None, None
        
        # Filter data within range
        df = df[(df['date'] >= start) & (df['date'] <= end)].copy()
        if len(df) < 1:  # Relaxed to 1 row minimum
            return None, None, None, None
        
        # Calculate profit/loss for buying each day
        end_price = df['close_price'].iloc[-1]
        df['profit_loss'] = (end_price - df['close_price']) * (amount / df['close_price'])
        df['shares_bought'] = amount / df['close_price']
        
        # Total profit/loss if bought every day
        total_invested = amount * len(df)
        total_profit_loss = df['profit_loss'].sum()
        
        # Overall percentage change (handle single-day case)
        start_price = df['close_price'].iloc[0]
        pct_change = ((end_price - start_price) / start_price) * 100 if len(df) > 1 else 0
        
        return df, total_profit_loss, pct_change, total_invested

    # Button and Results
    if st.button("Calculate Performance") and stock_symbol:
        with st.spinner("Fetching and analyzing data..."):
            start = start_date
            end = end_date
            
            if start > end:
                st.error("Start date must be before or equal to end date.")
            else:
                try:
                    df = fetch_stock_data_range(stock_symbol, start, end)
                    if df is not None:
                        results_df, total_profit_loss, pct_change, total_invested = calculate_performance(df, start, end, investment_amount)
                        
                        if results_df is not None:
                            currency = "₹" if market_choice == "Indian Stock Market" else "$"
                            
                            # Display Overall Results
                            st.subheader(f"Performance Summary for {stock_symbol}")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Total Invested", f"{currency}{total_invested:,.2f}")
                                st.metric("Total Profit/Loss", f"{currency}{total_profit_loss:,.2f}", 
                                        delta_color="normal" if total_profit_loss >= 0 else "inverse")
                            with col2:
                                st.metric("Percentage Change", f"{pct_change:.2f}%", 
                                        delta_color="normal" if pct_change >= 0 else "inverse")
                                st.write(f"From {start} to {end}")

                            # Detailed Table
                            st.subheader("Daily Investment Outcomes")
                            display_df = results_df[['date', 'close_price', 'shares_bought', 'profit_loss']]
                            display_df.columns = ['Date', 'Buy Price', 'Shares Bought', 'Profit/Loss']
                            display_df['Profit/Loss'] = display_df['Profit/Loss'].apply(lambda x: f"{currency}{x:,.2f}")
                            display_df['Buy Price'] = display_df['Buy Price'].apply(lambda x: f"{currency}{x:,.2f}")
                            st.dataframe(display_df.style.format({"Shares Bought": "{:.4f}"}))

                            # Chart
                            st.subheader("Price Trend and Profit/Loss")
                            fig, ax1 = plt.subplots(figsize=(10, 5))
                            
                            # Plot stock price
                            ax1.plot(results_df['date'], results_df['close_price'], label="Close Price", 
                                    color="#1f77b4", linewidth=2)
                            ax1.set_xlabel("Date", fontsize=12)
                            ax1.set_ylabel(f"Price ({currency})", fontsize=12, color="#1f77b4")
                            ax1.tick_params(axis='y', labelcolor="#1f77b4")
                            ax1.grid(True, linestyle='--', alpha=0.7)
                            
                            # Plot profit/loss on secondary axis
                            ax2 = ax1.twinx()
                            ax2.plot(results_df['date'], results_df['profit_loss'], label="Profit/Loss", 
                                    color="#ff7f0e", linewidth=2, linestyle='--')
                            ax2.set_ylabel(f"Profit/Loss ({currency})", fontsize=12, color="#ff7f0e")
                            ax2.tick_params(axis='y', labelcolor="#ff7f0e")
                            
                            # Titles and legend
                            plt.title(f"{stock_symbol} Performance: {start} to {end}", fontsize=14, fontweight='bold')
                            fig.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2)
                            plt.xticks(rotation=45)
                            fig.tight_layout()
                            st.pyplot(fig)
                        else:
                            st.error("Not enough data points between the selected dates (minimum 1 required).")
                    else:
                        st.error("No data available for this symbol or date range.")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {str(e)}")
                    
# --- Feature: Portfolio Optimizer ---
elif selection == "Portfolio Optimizer":
    st.header("📊 Portfolio Optimizer")

    # Input Section
    with st.container():
        st.subheader("Investment Inputs")
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Select Indian Stocks**")
            indian_selected = st.multiselect("Indian Stocks", options=indian_tickers, key="indian_stocks")
            st.write("**Select US Stocks**")
            us_selected = st.multiselect("US Stocks", options=us_tickers, key="us_stocks")

        with col2:
            investment_amount = st.number_input("Total Investment Amount ($)", min_value=1000.0, value=100000.0, step=1000.0)
            risk_tolerance = st.selectbox("Risk Tolerance Level", ["Low", "Medium", "High"])
            optimization_method = st.radio("Optimization Method", ["Traditional (MPT)", "Machine Learning"])

    # Combine selected stocks
    selected_stocks = indian_selected + us_selected
    stock_market_map = {stock: "Indian" for stock in indian_selected}
    stock_market_map.update({stock: "US" for stock in us_selected})

    # Fetch financial metrics and calculate Alpha
    @st.cache_data
    def fetch_financial_metrics(selected_stocks, stock_market_map):
        financial_data = {}
        risk_free_rate = 0.04
        for stock in selected_stocks:
            suffix = ".NS" if stock_market_map[stock] == "Indian" else ""
            ticker = yf.Ticker(f"{stock}{suffix}")
            try:
                info = ticker.info
                hist = ticker.history(period="1y")
                benchmark_ticker = yf.Ticker("^NSEI" if stock_market_map[stock] == "Indian" else "^GSPC")
                bench_hist = benchmark_ticker.history(period="1y")
                if not hist.empty and not bench_hist.empty:
                    aligned_data = pd.concat([hist['Close'], bench_hist['Close']], axis=1, keys=['Stock', 'Benchmark']).dropna()
                    stock_returns = aligned_data['Stock'].pct_change().dropna()
                    bench_returns = aligned_data['Benchmark'].pct_change().dropna()
                    cov = np.cov(stock_returns, bench_returns)[0, 1]
                    bench_var = np.var(bench_returns)
                    beta = cov / bench_var if bench_var != 0 else info.get('beta', np.nan)
                    annual_stock_return = stock_returns.mean() * 252
                    annual_bench_return = bench_returns.mean() * 252
                    alpha = annual_stock_return - (risk_free_rate + beta * (annual_bench_return - risk_free_rate))
                else:
                    alpha = np.nan
                    beta = info.get('beta', np.nan)

                financials = ticker.financials
                fcf = financials.loc['Free Cash Flow'].iloc[0] if 'Free Cash Flow' in financials.index and not pd.isna(financials.loc['Free Cash Flow'].iloc[0]) else np.nan

                financial_data[stock] = {
                    'Alpha': alpha * 100,
                    'Beta': beta,
                    'P/E Ratio': info.get('trailingPE', np.nan),
                    'Market Cap': info.get('marketCap', np.nan),
                    'Dividend Yield': info.get('dividendYield', 0.0) * 100,
                    'Debt-to-Equity': info.get('debtToEquity', np.nan),
                    'EPS': info.get('trailingEps', np.nan),
                    'ROE': info.get('returnOnEquity', np.nan) * 100,
                    'Free Cash Flow': fcf,
                    'P/B Ratio': info.get('priceToBook', np.nan),
                    'Current Ratio': info.get('currentRatio', np.nan)
                }
            except Exception:
                financial_data[stock] = {
                    'Alpha': np.nan,
                    'Beta': np.nan,
                    'P/E Ratio': np.nan,
                    'Market Cap': np.nan,
                    'Dividend Yield': 0.0,
                    'Debt-to-Equity': np.nan,
                    'EPS': np.nan,
                    'ROE': np.nan,
                    'Free Cash Flow': np.nan,
                    'P/B Ratio': np.nan,
                    'Current Ratio': np.nan
                }
        return pd.DataFrame(financial_data).T

    @st.cache_data
    def prepare_data_for_ml(selected_stocks, stock_market_map, financial_df):
        combined_df = pd.DataFrame()
        for stock in selected_stocks:
            if stock in indian_tickers:
                df = indian_df[indian_df['Ticker'] == stock][['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].copy()
            else:
                df = us_df[us_df['Ticker'] == stock][['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].copy()
            combined_df = pd.concat([combined_df, df])

        pivot_close = combined_df.pivot(index='Date', columns='Ticker', values='Close').dropna()
        daily_returns = pivot_close.pct_change().dropna()

        features_list = []
        for stock in selected_stocks:
            stock_data = combined_df[combined_df['Ticker'] == stock].set_index('Date')
            stock_data['Returns'] = stock_data['Close'].pct_change()
            stock_data['Volatility'] = stock_data['Returns'].rolling(window=20).std() * np.sqrt(252)
            stock_data['RSI'] = ta.momentum.RSIIndicator(stock_data['Close']).rsi()
            stock_data['MACD'] = ta.trend.MACD(stock_data['Close']).macd()
            stock_data['SMA_20'] = ta.trend.SMAIndicator(stock_data['Close'], window=20).sma_indicator()
            stock_data['Volume_Change'] = stock_data['Volume'].pct_change()
            for metric in ['Alpha', 'Beta', 'P/E Ratio', 'Market Cap', 'Dividend Yield', 'Debt-to-Equity',
                          'EPS', 'ROE', 'Free Cash Flow', 'P/B Ratio', 'Current Ratio']:
                stock_data[metric] = financial_df.loc[stock, metric]
            features_list.append(stock_data[['Returns', 'Volatility', 'RSI', 'MACD', 'SMA_20', 'Volume_Change',
                                            'Alpha', 'Beta', 'P/E Ratio', 'Market Cap', 'Dividend Yield',
                                            'Debt-to-Equity', 'EPS', 'ROE', 'Free Cash Flow', 'P/B Ratio',
                                            'Current Ratio']])

        features_df = pd.concat(features_list, axis=1, keys=selected_stocks)
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        technical_columns = [(stock, col) for stock in selected_stocks for col in ['Returns', 'Volatility', 'RSI', 'MACD', 'SMA_20', 'Volume_Change']]
        financial_columns = [(stock, col) for stock in selected_stocks for col in ['Alpha', 'Beta', 'P/E Ratio', 'Market Cap', 'Dividend Yield',
                                                                                'Debt-to-Equity', 'EPS', 'ROE', 'Free Cash Flow', 'P/B Ratio', 'Current Ratio']]
        features_df[technical_columns] = features_df[technical_columns].clip(lower=-1e5, upper=1e5).fillna(0)
        features_df[financial_columns] = features_df[financial_columns].clip(lower=-1e10, upper=1e10).fillna(features_df[financial_columns].median())
        features_df = features_df.dropna(how='all')
        if not np.all(np.isfinite(features_df)):
            features_df = features_df.fillna(0)

        return daily_returns, features_df

    @st.cache_data
    def train_ml_model(features_df, daily_returns, selected_stocks, risk_tolerance):
        if features_df.empty or len(features_df) < 10 or not np.all(np.isfinite(features_df)):
            st.warning("Invalid or insufficient data for ML training. Falling back to Traditional (MPT) weights.")
            pivot_df, daily_returns, mean_returns, cov_matrix = get_traditional_returns(selected_stocks, stock_market_map)
            result = optimize_portfolio(mean_returns, cov_matrix, financial_df, selected_stocks, risk_tolerance)
            if result.success:
                return None, None, result.x
            else:
                raise ValueError("Fallback MPT optimization failed.")

        mean_returns = daily_returns.mean()
        cov_matrix = daily_returns.cov()
        num_assets = len(selected_stocks)
        init_guess = [1 / num_assets] * num_assets
        bounds = [(0, 1)] * num_assets
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

        def neg_sharpe(weights, risk_penalty=1.0):
            port_return = np.dot(weights, mean_returns) * 252
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
            sharpe = (port_return - 0.04) / port_vol
            return -sharpe * risk_penalty

        risk_penalties = {'Low': 1.5, 'Medium': 1.0, 'High': 0.8}
        result = minimize(neg_sharpe, init_guess, args=(risk_penalties[risk_tolerance],),
                         method='SLSQP', bounds=bounds, constraints=constraints)
        optimal_weights = result.x if result.success else np.array(init_guess)

        X = features_df.values
        y = np.tile(optimal_weights, (X.shape[0], 1))
        test_size = min(0.2, max(1 / len(X), 0.1)) if len(X) > 1 else 0.1
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)

        return model, scaler, optimal_weights

    @st.cache_data
    def get_traditional_returns(selected_stocks, stock_market_map):
        combined_df = pd.DataFrame()
        for stock in selected_stocks:
            if stock in indian_tickers:
                df = indian_df[indian_df['Ticker'] == stock][['Date', 'Ticker', 'Adj Close']].copy()
            else:
                df = us_df[us_df['Ticker'] == stock][['Date', 'Ticker', 'Adj Close']].copy()
            combined_df = pd.concat([combined_df, df])
        
        pivot_df = combined_df.pivot(index='Date', columns='Ticker', values='Adj Close').dropna()
        daily_returns = pivot_df.pct_change().dropna()
        mean_returns = daily_returns.mean()
        cov_matrix = daily_returns.cov()
        return pivot_df, daily_returns, mean_returns, cov_matrix

    def optimize_portfolio(mean_returns, cov_matrix, financial_df, selected_stocks, risk_tolerance, risk_free_rate=0.04 / 252):
        num_assets = len(mean_returns)
        init_guess = [1 / num_assets] * num_assets
        bounds = [(0, 1)] * num_assets
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

        def neg_sharpe(weights):
            port_return = np.dot(weights, mean_returns)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe = (port_return - risk_free_rate) / port_vol
            
            penalty = 0
            for i, stock in enumerate(selected_stocks):
                alpha = financial_df.loc[stock, 'Alpha']
                beta = financial_df.loc[stock, 'Beta']
                pe = financial_df.loc[stock, 'P/E Ratio']
                div_yield = financial_df.loc[stock, 'Dividend Yield']
                debt_eq = financial_df.loc[stock, 'Debt-to-Equity']
                eps = financial_df.loc[stock, 'EPS']
                roe = financial_df.loc[stock, 'ROE']
                fcf = financial_df.loc[stock, 'Free Cash Flow']
                pb = financial_df.loc[stock, 'P/B Ratio']
                current = financial_df.loc[stock, 'Current Ratio']
                
                if not np.isnan(alpha):
                    penalty -= weights[i] * (alpha / 5)
                if not np.isnan(beta):
                    penalty += weights[i] * (beta / 2)
                if not np.isnan(pe):
                    penalty += weights[i] * (pe / 50)
                if not np.isnan(div_yield):
                    penalty -= weights[i] * (div_yield / 5)
                if not np.isnan(debt_eq):
                    penalty += weights[i] * (debt_eq / 100)
                if not np.isnan(eps):
                    penalty -= weights[i] * (eps / 10)
                if not np.isnan(roe):
                    penalty -= weights[i] * (roe / 20)
                if not np.isnan(fcf):
                    penalty -= weights[i] * (fcf / 1e9)
                if not np.isnan(pb):
                    penalty += weights[i] * (pb / 5)
                if not np.isnan(current):
                    penalty -= weights[i] * (current / 2)

            risk_factors = {'Low': 2.0, 'Medium': 1.0, 'High': 0.5}
            penalty *= risk_factors[risk_tolerance]
            
            return -(sharpe - 0.1 * penalty)

        result = minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        return result

    # Button and Results
    if st.button("Run Portfolio Optimization") and len(selected_stocks) >= 2:
        with st.spinner("Optimizing portfolio..."):
            try:
                financial_df = fetch_financial_metrics(selected_stocks, stock_market_map)

                if optimization_method == "Machine Learning":
                    daily_returns, features_df = prepare_data_for_ml(selected_stocks, stock_market_map, financial_df)
                    model, scaler, fallback_weights = train_ml_model(features_df, daily_returns, selected_stocks, risk_tolerance)
                    
                    if model is None:
                        weights = fallback_weights
                    else:
                        latest_features = features_df.iloc[-1].values.reshape(1, -1)
                        latest_features_scaled = scaler.transform(latest_features)
                        predicted_weights = model.predict(latest_features_scaled)[0]
                        predicted_weights = np.clip(predicted_weights, 0, 1)
                        predicted_weights /= predicted_weights.sum() if predicted_weights.sum() > 0 else 1.0
                        weights = predicted_weights

                    mean_returns = daily_returns.mean()
                    cov_matrix = daily_returns.cov()
                    annual_return = np.dot(weights, mean_returns) * 252
                    annual_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
                    sharpe_ratio = (annual_return - 0.04) / annual_volatility
                else:
                    pivot_df, daily_returns, mean_returns, cov_matrix = get_traditional_returns(selected_stocks, stock_market_map)
                    result = optimize_portfolio(mean_returns, cov_matrix, financial_df, selected_stocks, risk_tolerance)
                    if result.success:
                        weights = result.x
                        annual_return = np.dot(weights, mean_returns) * 252
                        annual_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
                        sharpe_ratio = (annual_return - 0.04) / annual_volatility
                    else:
                        raise ValueError("Traditional optimization failed.")

                st.subheader("📌 Optimized Portfolio Summary")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Expected Annual Return", f"{annual_return * 100:.2f}%")
                    st.metric("Expected Volatility", f"{annual_volatility * 100:.2f}%")
                with col2:
                    st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
                    st.metric("Stocks Selected", len(selected_stocks))

                st.subheader("📊 Allocations")
                alloc_df = pd.DataFrame({
                    'Stock': selected_stocks,
                    'Market': [stock_market_map[stock] for stock in selected_stocks],
                    'Weight %': (weights * 100).round(2),
                    'Investment Amount': (weights * investment_amount).round(2)
                })
                alloc_df['Investment Amount'] = alloc_df.apply(
                    lambda row: f"₹{row['Investment Amount']*75:,.2f}" if row['Market'] == "Indian" else f"${row['Investment Amount']:,.2f}",
                    axis=1
                )
                st.dataframe(alloc_df)

                st.subheader("📈 Financial Metrics of Selected Stocks")
                financial_display = financial_df.copy()
                financial_display['Alpha'] = financial_display['Alpha'].round(2)
                financial_display['Beta'] = financial_display['Beta'].round(2)
                financial_display['P/E Ratio'] = financial_display['P/E Ratio'].round(2)
                financial_display['Market Cap'] = financial_display.apply(
                    lambda row: f"${row['Market Cap']/1e9:,.2f}B" if not np.isnan(row['Market Cap']) else "N/A", axis=1)
                financial_display['Dividend Yield'] = financial_display['Dividend Yield'].round(2)
                financial_display['Debt-to-Equity'] = financial_display['Debt-to-Equity'].round(2)
                financial_display['EPS'] = financial_display['EPS'].round(2)
                financial_display['ROE'] = financial_display['ROE'].round(2)
                financial_display['Free Cash Flow'] = financial_display.apply(
                    lambda row: f"${row['Free Cash Flow']/1e6:,.2f}M" if not np.isnan(row['Free Cash Flow']) else "N/A", axis=1)
                financial_display['P/B Ratio'] = financial_display['P/B Ratio'].round(2)
                financial_display['Current Ratio'] = financial_display['Current Ratio'].round(2)
                st.dataframe(financial_display)

                st.subheader("📈 Portfolio Allocation")
                import matplotlib.pyplot as plt
                import seaborn as sns
                fig, ax = plt.subplots(figsize=(12, 6), dpi=120)
                colors = sns.color_palette("Spectral", len(selected_stocks))
                wedges, texts, autotexts = ax.pie(
                    weights,
                    labels=None,
                    autopct='%1.1f%%',
                    startangle=90,
                    colors=colors,
                    wedgeprops=dict(edgecolor='white', linewidth=4, alpha=0.9),
                    textprops=dict(fontsize=14, fontfamily="Roboto", fontweight="bold"),
                    explode=[0.15 if w == max(weights) else 0.05 for w in weights],
                    shadow=True
                )
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                ax.add_artist(plt.Circle((0,0), 0.4, color='white'))  # Donut effect
                ax.axis('equal')
                ax.legend(
                    wedges,
                    [f"{s} ({w*100:.1f}%)" for s, w in zip(selected_stocks, weights)],
                    title="Stocks",
                    loc="center left",
                    bbox_to_anchor=(1, 0, 0.5, 1),
                    fontsize=12,
                    title_fontsize=16,
                    frameon=True,
                    facecolor='white',
                    edgecolor='black'
                )
                ax.set_facecolor('#E6E6EB')
                fig.patch.set_facecolor('#E6E6EB')
                fig.suptitle("Portfolio Allocation", fontsize=20, fontweight='bold', fontfamily="Roboto")
                plt.text(0, -1.2, f"Total Investment: ${investment_amount:,.2f}", 
                         ha='center', fontsize=12, fontfamily="Roboto", fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)

                st.subheader("🤖 Portfolio Summary (AI-Generated)")
                portfolio_details = {
                    "Stocks": selected_stocks,
                    "Markets": [stock_market_map[stock] for stock in selected_stocks],
                    "Weights": [f"{w*100:.2f}%" for w in weights],
                    "Expected Annual Return": f"{annual_return * 100:.2f}%",
                    "Expected Volatility": f"{annual_volatility * 100:.2f}%",
                    "Sharpe Ratio": f"{sharpe_ratio:.2f}",
                    "Risk Tolerance": risk_tolerance,
                    "Method": optimization_method,
                    "Financial Metrics": financial_df.to_dict()
                }
                prompt = (
                    f"Generate a concise summary of the following portfolio optimization results in 6-7 lines: {portfolio_details}. "
                    f"The portfolio was optimized using {optimization_method} with a {risk_tolerance} risk tolerance, "
                    f"considering financial metrics (Alpha, Beta, P/E Ratio, Market Cap, Dividend Yield, Debt-to-Equity, EPS, ROE, "
                    f"Free Cash Flow, P/B Ratio, Current Ratio). Highlight key strengths (e.g., diversification, high Sharpe Ratio) "
                    f"and potential risks (e.g., currency risk, volatility). Mention the expected annual return, volatility, and Sharpe Ratio. "
                    f"Do not suggest further analysis of financial metrics or debt-to-equity ratios. Keep the tone professional and succinct."
                )
                try:
                    model = genai.GenerativeModel("gemini-1.5-flash-latest")
                    response = model.generate_content(prompt)
                    st.write(response.text)
                except Exception as e:
                    st.warning(f"Unable to generate AI summary: {str(e)}. Showing portfolio metrics only.")

            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")
    else:
        if len(selected_stocks) < 2:
            st.info("👆 Select at least two stocks (from either or both markets) to perform portfolio optimization.")


# --- Feature: Best Buy/Sell Timing ---
elif selection == "Best Buy/Sell Timing":
    st.header("📅 Best Buy/Sell Timing Analyzer")

    # Input Section
    with st.container():
        st.subheader("Select Stock and Date Range")
        col1, col2 = st.columns(2)

        with col1:
            market_choice = st.radio("Select Market:", ["Indian Stock Market", "US Stock Market"])
            stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL for US, RELIANCE for India):")

        with col2:
            today = pd.to_datetime("2025-04-19")  # Current date as per instructions
            start_date = st.date_input("Start Date", value=pd.to_datetime("2024-01-01"), max_value=today)
            end_date = st.date_input("End Date", value=today, max_value=today)
            currency = "₹" if market_choice == "Indian Stock Market" else "$"
            investment_amount = st.number_input(f"Investment Amount ({currency})", min_value=1.0, value=1000.0, step=100.0)

    # Function to fetch stock data
    @st.cache_data
    def fetch_stock_data_range(symbol, start, end):
        suffix = ".NS" if market_choice == "Indian Stock Market" else ""
        try:
            stock = yf.Ticker(f"{symbol}{suffix}")
            df = stock.history(start=start, end=end + pd.Timedelta(days=1))  # Include end date
            if df.empty:
                st.warning(f"No data returned for {symbol} between {start} and {end}.")
                return None
            df.reset_index(inplace=True)
            df = df[['Date', 'Close']]
            df.rename(columns={'Date': 'date', 'Close': 'close_price'}, inplace=True)
            df['date'] = pd.to_datetime(df['date']).dt.date
            return df
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")
            return None

    # Function to find best buy/sell points and calculate investment outcome
    def find_best_buy_sell(df, investment_amount):
        if df is None or df.empty:
            return None, None, None, None, None, None, None
        
        # Find the lowest and highest closing prices
        buy_idx = df['close_price'].idxmin()
        sell_idx = df['close_price'].idxmax()
        
        best_buy = {
            'date': df.loc[buy_idx, 'date'],
            'price': df.loc[buy_idx, 'close_price']
        }
        best_sell = {
            'date': df.loc[sell_idx, 'date'],
            'price': df.loc[sell_idx, 'close_price']
        }
        
        # Calculate potential profit/loss
        profit_loss = best_sell['price'] - best_buy['price']
        profit_pct = (profit_loss / best_buy['price']) * 100 if best_buy['price'] != 0 else 0
        
        # Calculate investment outcome
        shares_bought = investment_amount / best_buy['price'] if best_buy['price'] != 0 else 0
        value_at_sell = shares_bought * best_sell['price']
        investment_profit_loss = value_at_sell - investment_amount
        investment_profit_pct = (investment_profit_loss / investment_amount) * 100 if investment_amount != 0 else 0
        
        return best_buy, best_sell, profit_loss, profit_pct, shares_bought, value_at_sell, investment_profit_loss, investment_profit_pct

    # Button and Results
    if st.button("Analyze Buy/Sell Timing") and stock_symbol:
        with st.spinner("Fetching and analyzing data..."):
            start = start_date
            end = end_date
            
            if start > end:
                st.error("Start date must be before or equal to end date.")
            else:
                try:
                    df = fetch_stock_data_range(stock_symbol, start, end)
                    if df is not None:
                        best_buy, best_sell, profit_loss, profit_pct, shares_bought, value_at_sell, investment_profit_loss, investment_profit_pct = find_best_buy_sell(df, investment_amount)
                        
                        if best_buy and best_sell:
                            # Display Results
                            st.subheader(f"Best Buy/Sell Timing for {stock_symbol}")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Best Time to Buy", f"{currency}{best_buy['price']:,.2f}")
                                st.write(f"Date: {best_buy['date']}")
                            with col2:
                                st.metric("Best Time to Sell", f"{currency}{best_sell['price']:,.2f}")
                                st.write(f"Date: {best_sell['date']}")
                            
                            st.metric("Potential Profit/Loss per Share", f"{currency}{profit_loss:,.2f}", 
                                     delta=f"{profit_pct:.2f}%", 
                                     delta_color="normal" if profit_loss >= 0 else "inverse")
                            
                            # Investment Outcome
                            st.subheader(f"Investment Outcome for {currency}{investment_amount:,.2f}")
                            col3, col4 = st.columns(2)
                            with col3:
                                st.metric("Shares Purchased", f"{shares_bought:,.4f}")
                                st.metric("Value at Sell Date", f"{currency}{value_at_sell:,.2f}")
                            with col4:
                                st.metric("Investment Profit/Loss", f"{currency}{investment_profit_loss:,.2f}", 
                                         delta=f"{investment_profit_pct:.2f}%", 
                                         delta_color="normal" if investment_profit_loss >= 0 else "inverse")
                            
                            # Chart
                            st.subheader("Price Trend with Buy/Sell Points")
                            fig, ax = plt.subplots(figsize=(10, 5))
                            
                            # Plot stock price
                            ax.plot(df['date'], df['close_price'], label="Close Price", 
                                    color="#1f77b4", linewidth=2)
                            
                            # Highlight buy/sell points
                            ax.scatter([best_buy['date']], [best_buy['price']], 
                                      color='green', s=100, label="Best Buy", marker='^')
                            ax.scatter([best_sell['date']], [best_sell['price']], 
                                      color='red', s=100, label="Best Sell", marker='v')
                            
                            ax.set_xlabel("Date", fontsize=12)
                            ax.set_ylabel(f"Price ({currency})", fontsize=12)
                            ax.set_title(f"{stock_symbol} Price Trend: {start} to {end}", 
                                        fontsize=14, fontweight='bold')
                            ax.grid(True, linestyle='--', alpha=0.7)
                            ax.legend(loc='best')
                            plt.xticks(rotation=45)
                            fig.tight_layout()
                            st.pyplot(fig)
                            
                            # Note
                            st.info("Note: The best buy/sell points are based on the lowest and highest closing prices in the selected period. The investment outcome assumes buying on the best buy date and selling on the best sell date, excluding transaction costs or taxes.")
                        else:
                            st.error("Not enough data to determine buy/sell points.")
                    else:
                        st.error("No data available for this symbol or date range.")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {str(e)}")