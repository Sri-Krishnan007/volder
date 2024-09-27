import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
from sklearn.ensemble import IsolationForest
import numpy as np
import mysql.connector
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta

# Database connection
def connect_db():
    return mysql.connector.connect(
        host='localhost',
        user='root',
        password='krish2705',
        database='chatbot'
    )

def fetch_common_qa():
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT question, answer FROM dataas")
    qa_pairs = cursor.fetchall()
    conn.close()
    return qa_pairs

def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    return " ".join(tokens)

def fetch_stock_data(symbol, period='1d', interval='5m'):
    stock = yf.Ticker(symbol)
    df = stock.history(period=period, interval=interval)
    return df

def create_candlestick_plot(stock_data):
    time = stock_data.index
    open_price = stock_data['Open']
    high_price = stock_data['High']
    low_price = stock_data['Low']
    close_price = stock_data['Close']
    volume = stock_data['Volume']
    
    fig = go.Figure(data=[go.Candlestick(
        x=time,
        open=open_price,
        high=high_price,
        low=low_price,
        close=close_price,
        increasing_line_color='green',
        decreasing_line_color='red',
        name='Candlestick'
    )])
    
    latest_time = time[-1]
    latest_close = close_price[-1]
    latest_volume = volume[-1]
    
    fig.add_annotation(
        x=latest_time, y=latest_close,
        text=f"Volume: {latest_volume}",
        showarrow=True,
        arrowhead=2,
        ax=-40, ay=-40,
        font=dict(size=12, color="red")
    )
    
    fig.update_layout(
        title="Real-Time Candlestick Chart with Volume",
        xaxis_title='Time',
        yaxis_title='Stock Price',
        xaxis_rangeslider_visible=False,
    )
    
    return fig

def detect_anomalies(data):
    iso_forest = IsolationForest(contamination=0.01)
    data['Anomaly'] = iso_forest.fit_predict(data[['Volume']])
    return data



def fetch_data(ticker, period='2d', interval='15m'):
    data = yf.download(ticker, period=period, interval=interval)
    data.reset_index(inplace=True)
    return data
def calculate_indicators(df):
    # Calculate Simple Moving Averages (SMA)
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # Calculate Exponential Moving Averages (EMA)
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    
    # Calculate RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

def identify_patterns(df):
    df = calculate_indicators(df)
    
    # Initialize pattern columns
    df['Continuation'] = 0
    df['Reversal'] = 0
    df['Bilateral'] = 0
    
    # Identify patterns
    # Example patterns:
    
    # Continuation Pattern: Bullish Flag (price breaks above SMA_20)
    df.loc[(df['Close'] > df['SMA_20']) & (df['Close'] > df['EMA_20']), 'Continuation'] = 1
    
    # Reversal Pattern: Head and Shoulders (simple heuristic based on SMA cross)
    df.loc[(df['SMA_20'] < df['SMA_50']) & (df['RSI'] < 30), 'Reversal'] = 1
    
    # Bilateral Pattern: Symmetrical Triangle (price within a range)
    price_range = df['Close'].rolling(window=20).max() - df['Close'].rolling(window=20).min()
    df.loc[(price_range < df['Close'].rolling(window=20).mean() * 0.05), 'Bilateral'] = 1
    
    return df
def plot_candlestick(df):
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                                         open=df['Open'],
                                         high=df['High'],
                                         low=df['Low'],
                                         close=df['Close'],
                                         name='Candlestick'),
                          go.Scatter(x=df.index[df['Continuation'] == 1],
                                     y=df['Close'][df['Continuation'] == 1],
                                     mode='markers',
                                     marker=dict(color='blue', size=8),
                                     name='Continuation Pattern'),
                          go.Scatter(x=df.index[df['Reversal'] == 1],
                                     y=df['Close'][df['Reversal'] == 1],
                                     mode='markers',
                                     marker=dict(color='green', size=8),
                                     name='Reversal Pattern'),
                          go.Scatter(x=df.index[df['Bilateral'] == 1],
                                     y=df['Close'][df['Bilateral'] == 1],
                                     mode='markers',
                                     marker=dict(color='orange', size=8),
                                     name='Bilateral Pattern')
                         ])
    fig.update_layout(title='Candlestick Chart with Patterns',
                      xaxis_title='Date',
                      yaxis_title='Price')
    return fig
def find_best_answer(user_question):
    qa_pairs = fetch_common_qa()
    questions = [preprocess(q[0]) for q in qa_pairs]
    answers = [q[1] for q in qa_pairs]
    
    vectorizer = TfidfVectorizer()
    question_vectors = vectorizer.fit_transform(questions)
    
    user_vector = vectorizer.transform([preprocess(user_question)])
    similarities = cosine_similarity(user_vector, question_vectors)
    best_match_idx = similarities.argmax()
    return answers[best_match_idx]

def format_answer(answer, detailed):
    if detailed:
        return answer
    else:
        return answer.split('.')[0]  # Return the first sentence for concise responses

# Streamlit app
def main():
    st.sidebar.title('Navigation')
    selection = st.sidebar.radio("Go to", ["Volume Monitoring", "Pattern Detection", "Trading Chatbot","Sentiment Analyzer"])

    if selection == "Volume Monitoring":
        st.title('Stock Data Dashboard with AI-based Volume Spike Detection')
        
        companies = {
            'Reliance Industries': 'RELIANCE.NS',
            'Tata Consultancy Services': 'TCS.NS',
            'Infosys': 'INFY.NS',
            'HDFC Bank': 'HDFCBANK.NS',
            'State Bank of India': 'SBIN.BO'
        }
        
        selected_company = st.selectbox('Select a company', list(companies.keys()))
        stock_symbol = companies[selected_company]
        
        stock_data = fetch_stock_data(stock_symbol)
        fig = create_candlestick_plot(stock_data)
        st.plotly_chart(fig)
        
        stock_data['Volume_MA'] = stock_data['Volume'].rolling(window=10).mean()
        stock_data['Volume_STD'] = stock_data['Volume'].rolling(window=10).std()
        
        latest_volume = stock_data['Volume'].iloc[-1]
        ma = stock_data['Volume_MA'].iloc[-1]
        std = stock_data['Volume_STD'].iloc[-1]
        threshold = ma + 2 * std
        
        if latest_volume > threshold:
            st.warning(f"Volume Spike Detected: {latest_volume}! Threshold: {threshold:.2f}")
            st.audio("C:\\Users\\srikr\\Desktop\\COLLEGE\\Sem 5\\ST Joseph Hackathon\\mixkit-censorship-beep-1082.wav")
        else:
            st.success(f"No significant volume spike detected. Latest Volume: {latest_volume}, Threshold: {threshold:.2f}")
        
        stock_data = detect_anomalies(stock_data)
        
        if 'NS' in stock_symbol:
            st.write("The stock is listed on NSE.")
        elif 'BO' in stock_symbol:
            st.write("The stock is listed on BSE.")
        else:
            st.write("The stock is listed on multiple exchanges.")

    
    elif selection == "Pattern Detection":
        st.title('Advanced Stock Price Action Pattern Detection')
    
        ticker = st.text_input('Enter Stock Ticker', 'AAPL')
        period='2d'
        interval='15m'
    
        if st.button('Fetch Data and Detect Patterns'):
            st.write(f"Fetching data for {ticker} with {period} period and {interval} interval...")
            df = fetch_data(ticker, period, interval)
            df = identify_patterns(df)
        
            st.write("Detected Patterns:")
            for col in ['Continuation', 'Reversal', 'Bilateral']:
                if df[col].sum() > 0:
                    st.write(f"- {col} Pattern")

            fig = plot_candlestick(df)
            st.plotly_chart(fig)        

    elif selection == "Trading Chatbot":
        st.title('Trading Chatbot')
        user_question = st.text_input("Ask a trading-related question:")
        detailed = st.checkbox("Detailed answer")
        
        if user_question:
            answer = find_best_answer(user_question)
            formatted_answer = format_answer(answer, detailed)
            st.write(formatted_answer)
    elif selection == "Sentiment Analyzer":
        st.title("News Sentiment Analyzer")

        # API Key and Endpoint
        api_key = '2bd3bcdfa49d4fa08bf5fa744fb0dc3c'
        url = f'https://newsapi.org/v2/everything?q=stock&apiKey={api_key}'

        # Fetch News Data
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            news_articles = data['articles']
        else:
            st.error(f"Failed to fetch news. Status code: {response.status_code}")
            news_articles = []

        # Initialize Sentiment Analyzer
        analyzer = SentimentIntensityAnalyzer()

        # Set time filter (1 week ago)
        today = datetime.now()
        one_week_ago = today - timedelta(days=7)

        # Filter out old news
        filtered_news = [article for article in news_articles if datetime.strptime(article['publishedAt'], '%Y-%m-%dT%H:%M:%SZ') > one_week_ago]

        if not filtered_news:
            st.warning("No recent news found for the last week.")
        else:
            st.write(f"Found {len(filtered_news)} news articles from the past week.")

            # Function to analyze sentiment and generate buy/sell suggestion
            def get_sentiment_suggestion(article):
                sentiment = analyzer.polarity_scores(article['description'])
                if sentiment['compound'] > 0.05:
                    suggestion = "Buy"
                elif sentiment['compound'] < -0.05:
                    suggestion = "Sell"
                else:
                    suggestion = "Hold"
                return sentiment['compound'], suggestion

            # Process each filtered news article
            for article in filtered_news:
                title = article['title']
                description = article['description']
                published_at = article['publishedAt']
                
                sentiment_score, suggestion = get_sentiment_suggestion(article)
                
                st.subheader(f"News Title: {title}")
                st.write(f"Published At: {published_at}")
                st.write(f"Sentiment Score: {sentiment_score:.2f}")
                st.write(f"Suggestion: **{suggestion}**")
                st.write("-" * 50)        

if __name__ == "__main__":
    main()
