# **Cryptocurrency Price Prediction Using Media Sentiment Analysis**

## **Project Overview**

This project aims to predict cryptocurrency prices for the end of 2023 based on sentiment analysis of media coverage from 2021 to 2023. By leveraging machine learning models and utilizing historical price data alongside hourly news sentiment for major cryptocurrencies and related blockchain technologies, the project endeavors to understand the intricate relationship between news sentiment and cryptocurrency market trends.

The primary focus is on key cryptocurrencies such as Bitcoin and Ethereum, as well as topics like blockchain, altcoins, decentralized finance (DeFi), and non-fungible tokens (NFTs). Using the patterns learned from historical data, the models forecast price trends for the last two months of 2023 and evaluate their performance against actual market fluctuations.

---

## **Data Collection**

Data collection involves scraping historical cryptocurrency prices and news articles:

### **Cryptocurrency Price Data**
- Hourly price data for Bitcoin, Ethereum, and other major cryptocurrencies is fetched using the [CoinCap API](https://coincap.io/).
- Each data point includes:
  - Price in USD (`price_usd`)
  - 24-hour trading volume (`volume_usd`)
  - Market capitalization (`market_cap`)
  - Timestamps for alignment with news data.

### **News Data**
- News articles related to cryptocurrencies are collected hourly.
- Articles are processed for:
  - **Sentiment Analysis**: Polarity, subjectivity, and sentiment classification.
  - **Contextual Features**: Using embeddings to extract relationships between news content and price fluctuations.

---

## **Methodology**

### **1. Sentiment Analysis**
- The sentiment of news articles is determined using a transformer-based NLP model for classification:
  - **Sentiment Polarity**: Range from negative (-1) to positive (+1).
  - **Sentiment Class**: Labels such as negative, neutral, and positive.
- Cosine similarity is used to link relevant news articles to cryptocurrency price movements.

### **2. Feature Engineering**
- Lagging indicators (`price_usd_lag1`, `sentiment_polarity_lag1`, etc.) and moving averages (`price_usd_ma7`) are derived to capture temporal dependencies.
- Additional features such as volatility and rolling statistics enhance prediction accuracy.

### **3. Data Scaling**
- Features are normalized using `MinMaxScaler` to ensure effective model training and evaluation.

---

## **Model Development**

### **1. Long Short-Term Memory (LSTM)**
- A deep learning model optimized for sequential data.
- Captures dependencies between lagged features and sentiment data.
- Hyperparameters:
  - Layers: 1 LSTM layer with 50 units.
  - Optimizer: Adam.
  - Loss Function: Mean Squared Error (MSE).

### **2. Random Forest Regressor**
- A machine learning ensemble model for robust predictions.
- Handles feature interactions and non-linear relationships effectively.

---

## **Workflow**

1. **Data Preprocessing**
   - Data gaps are handled using interpolation for price and sentiment features.
   - News timestamps are aligned with price data using a nearest-neighbor approach.

2. **Model Training**
   - The dataset is split into training, validation, and testing sets:
     - **Training**: 2021-10-12 to 2022-12-31
     - **Validation**: 2023-01-01 to 2023-09-30
     - **Testing**: 2023-10-01 to 2023-12-19
   - Features and targets are scaled independently.

3. **Prediction**
   - Price predictions are generated using LSTM and Random Forest models.
   - Predictions are compared to actual prices using RMSE (Root Mean Square Error).

4. **Visualization**
   - Plots of actual vs. predicted prices for each category (e.g., Bitcoin, Ethereum) demonstrate model performance.

---

## **Technical Highlights**

- **Transformers**: Employed `SentenceTransformer` for contextual embeddings of news text.
- **Batch Processing**: Enabled efficient sentiment computation and feature extraction using tools like `tqdm`.
- **Scalable Models**: Combined deep learning (LSTM) with traditional machine learning (Random Forest).
- **Explainability**: Used cosine similarity and feature importance to assess the relevance of sentiment to price movements.

---

## **Key Results**

- **Accuracy**: Achieved RMSE values in the acceptable range for both LSTM and Random Forest models.
- **Insights**:
  - Positive sentiments tend to drive price increases, particularly during bullish market trends.
  - Sudden sentiment drops can precede market corrections.
- **Performance**: LSTM outperformed Random Forest for highly volatile assets, while Random Forest excelled with structured, historical data.

---

## **Usage**

### **Dependencies**
- Python 3.8+
- Libraries: 
  - `tensorflow`, `scikit-learn`, `transformers`, `pandas`, `numpy`, `matplotlib`, `tqdm`

### **Run the Code**
1. Clone the repository:
   ```bash
   git clone https://github.com/username/crypto_price_prediction.git
   cd crypto_price_prediction
    ```

2. Install dependencies: 
    ```bash
    pip install -r requirements.txt
    ```

3. Execute the main script: 
    ```bash
    python main.py
    ```

### **Future Work**
* Extend sentiment analysis to include additional factors like social media trends and developer activity.
* Explore alternative architectures such as Transformer models for improved sequential predictions.
* Incorporate real-time price forecasting with streaming data, if the data is available. 