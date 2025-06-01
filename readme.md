# 📈 Ultimate AI Trading Bot

The **Ultimate AI Trading Bot** is a powerful, real-time trading assistant designed to analyze markets using AI/ML, monitor financial news sentiment, generate trading signals with confidence scores, and send alerts directly to Discord. Built for traders who demand the edge in crypto, forex, stocks, and commodities.

---

## ⚙️ Features

- 🔮 **AI/ML-Powered Signal Detection**  
  Uses a trained machine learning model to predict buy/sell/neutral signals.

- 📰 **Live News Sentiment Analysis**  
  Integrates with NewsAPI and HuggingFace Transformers to extract sentiment from breaking news.

- 💬 **Discord Integration**  
  Sends alerts and confidence scores to your channel. High-confidence signals automatically mention the `@traders` role.

- 💹 **Real-Time Market Data**  
  Connects to TradingView WebSocket for live prices on major assets.

- 📊 **Custom Dashboard Commands**  
  Use `!signal` and `!chart` to view the latest signal and a historical confidence chart in Discord.

- 🧠 **Neutral-First Signal Priority**  
  Neutral signals are prioritized for review before buy/sell actions.

- 🔐 **Confidence Threshold Mentions**  
  Confidence scores > 75% will mention the assigned Discord role to highlight high-value opportunities.

- 💾 **Persistent Database Logging**  
  Logs all signals, confidence, sentiment, and news for historical reference using SQLite.

---

## 💼 Supported Markets

- **Cryptocurrency**: BTC, ETH, BNB, XRP, ADA, DOGE
- **Forex**: EUR/USD, GBP/USD, USD/JPY, AUD/USD, USD/CAD, USD/CHF, NZD/USD
- **Commodities**: Gold (XAU), Silver (XAG), USOIL, UKOIL
- **Stocks**: AAPL, TSLA, MSFT, AMZN, NVDA, BRK.B, JPM, V, META

---

## 🚀 Getting Started

### 1. Clone the Repo

```bash
git clone https://github.com/yourusername/ai-trading-bot.git
cd ai-trading-bot
