import streamlit as st
import yfinance as yf
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline
import plotly.express as px
import time
import nltk

# Download VADER once (quietly)
nltk.download('vader_lexicon', quiet=True)

st.title("🦅 Chad's Free US Stock News & Feelings Dashboard")

st.write("Type some American stock tickers like AAPL, TSLA, NVDA, MSFT and see news + if the news feels happy or sad!")

# Sidebar - easy controls
st.sidebar.header("Your Stock List")
tickers_input = st.sidebar.text_input("Tickers (comma separated)", "AAPL, TSLA, NVDA, MSFT, AMZN")
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

use_finbert = st.sidebar.checkbox("Use Smart Finance Brain (FinBERT - a bit slower but better)", value=False)

# Load the feeling checker
sia = SentimentIntensityAnalyzer()
finbert_pipe = None
if use_finbert:
    with st.spinner("Loading the smart finance brain... (only once)"):
        finbert_pipe = pipeline("text-classification", model="ProsusAI/finbert")

@st.cache_data(ttl=300)  # remembers for 5 minutes so it's fast and polite
def get_stock_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        news_list = stock.news[:12]  # get up to 12 recent news items
        return info, news_list
    except:
        return None, []

data = []

for ticker in tickers:
    with st.spinner(f"Checking {ticker}..."):
        info, news_list = get_stock_info(ticker)
        if not info:
            st.warning(f"Couldn't find {ticker} right now. Try again later!")
            continue
        
        sentiments = []
        for item in news_list:
            title = item.get('title', '')
            summary = item.get('summary', '') or ''
            text = title + ". " + summary
            
            if use_finbert and finbert_pipe:
                result = finbert_pipe(text)[0]
                score = 1 if result['label'] == 'positive' else -1 if result['label'] == 'negative' else 0
                label = result['label']
            else:
                score = sia.polarity_scores(text)['compound']
                label = "positive" if score > 0.05 else "negative" if score < -0.05 else "neutral"
            
            sentiments.append(score)
        
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
        
        data.append({
            'Ticker': ticker,
            'Price': round(info.get('currentPrice') or info.get('regularMarketPrice') or 0, 2),
            'Market Cap (Billions)': round((info.get('marketCap') or 0) / 1_000_000_000, 1),
            'P/E Ratio': round(info.get('trailingPE') or 0, 1),
            'Avg News Feeling': round(avg_sentiment, 2),
            'News Stories': len(news_list)
        })
        
        time.sleep(0.8)  # be nice to the free websites

df = pd.DataFrame(data)

if not df.empty:
    st.subheader("Your Stock Screen")
    st.dataframe(df.sort_values('Avg News Feeling', ascending=False), use_container_width=True)
    
    # Pretty bar chart of feelings
    fig = px.bar(df, x='Ticker', y='Avg News Feeling', 
                 color='Avg News Feeling',
                 color_continuous_scale='RdYlGn',
                 title="How Happy is the News for Each Stock?")
    st.plotly_chart(fig, use_container_width=True)

    # Show detailed news for one stock
    selected = st.selectbox("Click a stock to see its news and feelings", df['Ticker'])
    if selected:
        st.subheader(f"News for {selected}")
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
            
            st.markdown(f"**{feeling}** (score: {score:.2f}) — [{title}]({link})")
else:
    st.info("Add some tickers in the sidebar and press Enter!")

st.caption("100% Free • Made for fun • American stocks only • Refresh to get latest news")
