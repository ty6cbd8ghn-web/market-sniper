\import streamlit as st
import yfinance as yf
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline
import plotly.express as px
import nltk
from concurrent.futures import ThreadPoolExecutor

# Setup
st.set_page_config(page_title="Market Sniper", layout="wide", page_icon="📈")

nltk.download('vader_lexicon', quiet=True)

st.title("🦅 Market Sniper Dashboard")
st.caption("Live US stock sentiment + price action")

# Sidebar
st.sidebar.header("Controls")
tickers_input = st.sidebar.text_input(
    "Tickers (comma separated)",
    "AAPL, TSLA, NVDA, MSFT, AMZN"
)

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

use_finbert = st.sidebar.checkbox(
    "Use FinBERT (better but slower)", value=False
)

sniper_mode = st.sidebar.checkbox(
    "Sniper Mode (only strong signals)", value=False
)

# Load sentiment models
sia = SentimentIntensityAnalyzer()
finbert_pipe = None

if use_finbert:
    with st.spinner("Loading AI model..."):
        finbert_pipe = pipeline("text-classification", model="ProsusAI/finbert")

# Cache data
@st.cache_data(ttl=300)
def get_stock_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        return stock.info, stock.news[:10]
    except:
        return None, []

@st.cache_data(ttl=300)
def get_price_history(ticker):
    return yf.download(ticker, period="1mo", interval="1h")

# Signal logic
def get_signal(sentiment):
    if sentiment > 0.2:
        return "🚀 BUY"
    elif sentiment < -0.2:
        return "⚠️ SELL"
    else:
        return "⏸️ HOLD"

# Process each ticker
def process_ticker(ticker):
    info, news_list = get_stock_info(ticker)
    if not info:
        return None

    sentiments = []

    for item in news_list:
        text = (item.get('title', '') + ". " + (item.get('summary') or ''))

        if use_finbert and finbert_pipe:
            result = finbert_pipe(text)[0]
            score = 1 if result['label'] == 'positive' else -1 if result['label'] == 'negative' else 0
        else:
            score = sia.polarity_scores(text)['compound']

        sentiments.append(score)

    avg_sentiment = sum(sentiments)/len(sentiments) if sentiments else 0

    return {
        'Ticker': ticker,
        'Price': round(info.get('currentPrice') or info.get('regularMarketPrice') or 0, 2),
        'Market Cap (B)': round((info.get('marketCap') or 0)/1_000_000_000, 1),
        'P/E': round(info.get('trailingPE') or 0, 1),
        'Avg Sentiment': round(avg_sentiment, 2),
        'Signal': get_signal(avg_sentiment),
        'News Count': len(news_list)
    }

# Run processing
data = []
with ThreadPoolExecutor() as executor:
    results = list(executor.map(process_ticker, tickers))
    data = [r for r in results if r]

df = pd.DataFrame(data)

# Display results
if not df.empty:

    if sniper_mode:
        df = df[(df['Avg Sentiment'] > 0.25) | (df['Avg Sentiment'] < -0.25)]

    st.subheader("📊 Stock Overview")
    st.dataframe(df.sort_values('Avg Sentiment', ascending=False), use_container_width=True)

    # Top pick
    top = df.sort_values('Avg Sentiment', ascending=False).iloc[0]
    st.success(f"🔥 Top Play: {top['Ticker']} → {top['Signal']}")

    # Sentiment chart
    fig = px.bar(
        df,
        x='Ticker',
        y='Avg Sentiment',
        color='Avg Sentiment',
        color_continuous_scale='RdYlGn',
        title="News Sentiment Score"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Select ticker
    selected = st.selectbox("Select stock for details", df['Ticker'])

    # Price chart
    st.subheader(f"{selected} Price Chart")
    price_df = get_price_history(selected)

    if not price_df.empty:
        fig_price = px.line(price_df, x=price_df.index, y="Close",
                            title=f"{selected} Price (1 Month)")
        st.plotly_chart(fig_price, use_container_width=True)

    # News section
    st.subheader(f"📰 News for {selected}")
    _, news_list = get_stock_info(selected)

    for item in news_list:
        title = item.get('title', 'No title')
        link = item.get('link', '#')
        text = title + ". " + (item.get('summary') or '')

        if use_finbert and finbert_pipe:
            result = finbert_pipe(text)[0]
            feeling = result['label'].capitalize()
            score = result['score']
        else:
            score = sia.polarity_scores(text)['compound']
            feeling = "🟢 Positive" if score > 0.05 else "🔴 Negative" if score < -0.05 else "⚪ Neutral"

        st.markdown(f"**{feeling}** ({score:.2f}) — [{title}]({link})")

else:
    st.warning("No valid tickers found.")

st.caption("Free tool • Sentiment-based • Not financial advice")
