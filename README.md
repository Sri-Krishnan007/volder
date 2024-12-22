
# 📊 **Stock Monitoring and Trading Chatbot with Sentiment Analysis**

### 🚀 **Overview**

This Streamlit application integrates stock data visualization, pattern detection, anomaly detection, a chatbot for stock-related queries, and sentiment analysis of financial news. It leverages AI and ML models to provide insightful analytics and actionable insights for traders and stock enthusiasts.

---

## 📦 **Features**

1. **📈 Volume Monitoring**  
   - Real-time candlestick stock charts with volume analysis.  
   - AI-based anomaly detection using Isolation Forest.  
   - Alerts for significant volume spikes.

2. **🔍 Pattern Detection**  
   - Detect price action patterns (Continuation, Reversal, Bilateral).  
   - Interactive candlestick charts highlighting patterns.

3. **💬 Trading Chatbot**  
   - NLP-based chatbot for answering stock-related questions.  
   - Utilizes a database of common questions and answers.  
   - Provides concise or detailed responses.

4. **📰 Sentiment Analyzer**  
   - Fetch recent stock-related news articles.  
   - Perform sentiment analysis using Vader Sentiment Analyzer.  
   - Generate Buy/Hold/Sell suggestions based on sentiment scores.

---

## 🛠️ **Technologies Used**

- **Frontend**: Streamlit  
- **Backend**: Python  
- **Data Analysis**: Pandas, NumPy  
- **Visualization**: Plotly  
- **ML Models**: Isolation Forest  
- **NLP**: NLTK, TF-IDF Vectorization, Cosine Similarity  
- **Sentiment Analysis**: VaderSentiment  
- **Database**: MySQL  
- **API Integration**: NewsAPI, Yahoo Finance  

---

## ⚙️ **Setup and Installation**

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/your-username/stock-trading-chatbot.git
   cd stock-trading-chatbot
   ```

2. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Database Configuration**  
   - Ensure a MySQL database is set up with a table named `dataas`.  
   - Populate the table with common Q&A pairs.

4. **Run the Application**  
   ```bash
   streamlit run app.py
   ```

---


## 📊 **Usage**

1. **Volume Monitoring**  
   - Select a stock symbol from the dropdown.  
   - Monitor candlestick charts and volume anomalies.

2. **Pattern Detection**  
   - Enter a stock ticker symbol.  
   - Analyze detected patterns.

3. **Trading Chatbot**  
   - Ask trading-related questions and get intelligent responses.

4. **Sentiment Analyzer**  
   - Fetch recent stock news.  
   - Get sentiment-based buy/hold/sell recommendations.

---

## 🤝 **Contributing**

1. Fork the repository.  
2. Create a new branch (`git checkout -b feature-branch`).  
3. Commit your changes (`git commit -m "Add new feature"`).  
4. Push to the branch (`git push origin feature-branch`).  
5. Open a Pull Request.


## 📧 **Contact**

- **Author:** Sri Krishnan G  
- **Email:** [click here👇](srikrish2705guru@gmail.com)  

Happy Trading! 🚀📊
