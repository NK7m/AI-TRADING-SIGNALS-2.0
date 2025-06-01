# === ai_trading_bot.py ===
# Ultimate AI Trading Bot with market analysis, live news sentiment, trained ML signals, DB logging, Discord alerts, real TradingView WebSocket feed, and live dashboard commands

import asyncio
import datetime
import aiohttp
import json
import sqlite3
from discord.ext import commands, tasks
from transformers import pipeline
import websockets
import threading
import pandas as pd
import matplotlib.pyplot as plt
import io
import discord
import joblib
import numpy as np

# --- Discord Bot Setup ---
DISCORD_TOKEN = "YOUR_DISCORD_BOT_TOKEN"
CHANNEL_ID = 1234567890  # Replace with your channel ID
ROLE_ID = 987654321  # Replace with @traders role ID

bot = commands.Bot(command_prefix="!")

# --- Assets to Track ---
ASSETS = [
    # Crypto
    "BINANCE:BTCUSDT", "COINBASE:ETHUSD", "BINANCE:BNBUSDT", "BINANCE:XRPUSDT", "BINANCE:ADAUSDT", "BINANCE:DOGEUSDT",
    "BINANCE:MATICUSDT", "BINANCE:SOLUSDT", "BINANCE:AVAXUSDT", "BINANCE:DOTUSDT", "BINANCE:SHIBUSDT",

    # Forex
    "FX:EURUSD", "FX:GBPUSD", "FX:USDJPY", "FX:AUDUSD", "FX:USDCAD", "FX:USDCHF", "FX:NZDUSD", "FX:EURGBP", "FX:EURJPY",

    # Commodities
    "OANDA:XAUUSD", "OANDA:XAGUSD", "OANDA:USOIL", "OANDA:UKOIL", "OANDA:NATGASUSD", "OANDA:COPPER",

    # Stocks
    "NASDAQ:AAPL", "NASDAQ:TSLA", "NASDAQ:MSFT", "NASDAQ:AMZN", "NASDAQ:NVDA", "NYSE:BRK.B", "NYSE:JPM", "NYSE:V", "NASDAQ:META",
    "NASDAQ:GOOGL", "NASDAQ:NFLX", "NYSE:BABA", "NYSE:DIS", "NYSE:WMT"
]

# --- Signal Confidence Threshold ---
MENTION_THRESHOLD = 75

# --- Load Sentiment Pipeline ---
sentiment_pipeline = pipeline("sentiment-analysis")

# --- Load ML Model ---
ml_model = joblib.load("trained_model.pkl")  # Trained model file (e.g., random forest or SVM)

# --- Database Setup ---
conn = sqlite3.connect("signals.db", check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS signals (
    asset TEXT,
    signal TEXT,
    confidence INTEGER,
    sentiment TEXT,
    news TEXT,
    timestamp TEXT
)''')
conn.commit()

# --- News Scraper ---
async def fetch_news_headline(asset):
    query = asset.split(":")[-1]
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey=YOUR_NEWSAPI_KEY"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            data = await resp.json()
            if "articles" in data and data["articles"]:
                return data["articles"][0]["title"]
    return "No recent news"

# --- Sentiment Analysis ---
def analyze_sentiment(headline):
    result = sentiment_pipeline(headline)[0]
    return result["label"], headline

# --- Real-Time Market Feed (Finnhub WebSocket Example) ---
FINNHUB_API_KEY = "YOUR_FINNHUB_API_KEY"
latest_prices = {}

async def listen_tradingview():
    uri = f"wss://ws.finnhub.io?token={FINNHUB_API_KEY}"
    async with websockets.connect(uri) as websocket:
        for asset in ASSETS:
            symbol = asset.split(":")[-1]
            await websocket.send(json.dumps({"type": "subscribe", "symbol": symbol}))

        while True:
            response = await websocket.recv()
            data = json.loads(response)
            if data.get("type") == "trade":
                for trade in data["data"]:
                    latest_prices[trade["s"]] = {
                        "open": trade["p"],
                        "high": trade["p"],
                        "low": trade["p"],
                        "close": trade["p"],
                        "volume": trade["v"]
                    }

# --- ML Signal Classification ---
def classify_signal(price_data, sentiment_label):
    X = np.array([[price_data["open"], price_data["high"], price_data["low"], price_data["close"], price_data["volume"]]])
    signal_pred = ml_model.predict(X)[0]
    conf_pred = max(ml_model.predict_proba(X)[0]) * 100

    sentiment_boost = 10 if sentiment_label == "POSITIVE" else -10 if sentiment_label == "NEGATIVE" else 0
    adjusted_conf = int(min(max(conf_pred + sentiment_boost, 0), 100))

    return signal_pred.upper(), adjusted_conf

# --- Store Signal in Database ---
def store_signal(asset, signal, confidence, sentiment, news):
    timestamp = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    c.execute("INSERT INTO signals VALUES (?, ?, ?, ?, ?, ?)",
              (asset, signal, confidence, sentiment, news, timestamp))
    conn.commit()

# --- Discord Notification ---
async def send_signal(asset, signal, confidence, sentiment, news):
    channel = bot.get_channel(CHANNEL_ID)
    if channel is None:
        return

    message = (
        f"\n📊 **Signal Alert** — {asset}"
        f"\n🔍 Signal Type: {signal}"
        f"\n📈 Confidence: {confidence}%"
        f"\n📰 Sentiment: {sentiment} | \"{news}\""
        f"\n⏱️ Time: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"
    )

    if confidence >= MENTION_THRESHOLD:
        message += f"\n\n🔔 <@&{ROLE_ID}> High-confidence opportunity detected!"

    await channel.send(message)

# --- Periodic Market Scanner ---
@tasks.loop(minutes=1)
async def scan_markets():
    for asset in ASSETS:
        price_data = await fetch_price_data(asset)
        news = await fetch_news_headline(asset)
        sentiment_label, sentiment_text = analyze_sentiment(news)
        signal, confidence = classify_signal(price_data, sentiment_label)

        if signal == "NEUTRAL" or confidence >= 60:
            store_signal(asset, signal, confidence, sentiment_label, sentiment_text)
            await send_signal(asset, signal, confidence, sentiment_label, sentiment_text)

# --- Simulated Price Data (Fallback) ---
async def fetch_price_data(asset):
    symbol = asset.split(":")[-1]
    if symbol in latest_prices:
        return latest_prices[symbol]
    return {
        "open": 100 + hash(asset) % 50,
        "high": 150 + hash(asset) % 60,
        "low": 90 + hash(asset) % 40,
        "close": 110 + hash(asset) % 45,
        "volume": 1000 + hash(asset) % 5000
    }

# --- Command to Query Signals ---
@bot.command()
async def signal(ctx, asset: str):
    c.execute("SELECT * FROM signals WHERE asset=? ORDER BY timestamp DESC LIMIT 1", (asset.upper(),))
    row = c.fetchone()
    if row:
        await ctx.send(
            f"\n📊 **Latest Signal for {asset.upper()}**\n"
            f"🔍 Type: {row[1]}\n"
            f"📈 Confidence: {row[2]}%\n"
            f"📰 Sentiment: {row[3]}\n"
            f"📅 Timestamp: {row[5]}"
        )
    else:
        await ctx.send(f"No signal found for {asset.upper()}")

# --- Command: Show Asset History Chart ---
@bot.command()
async def chart(ctx, asset: str):
    c.execute("SELECT timestamp, confidence FROM signals WHERE asset=? ORDER BY timestamp DESC LIMIT 20", (asset.upper(),))
    rows = c.fetchall()
    if not rows:
        await ctx.send("No data found.")
        return

    df = pd.DataFrame(rows[::-1], columns=["timestamp", "confidence"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    plt.figure(figsize=(10, 4))
    plt.plot(df["timestamp"], df["confidence"], marker="o", label="Confidence")
    plt.title(f"{asset.upper()} Confidence History")
    plt.xlabel("Time")
    plt.ylabel("Confidence %")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    await ctx.send(file=discord.File(buf, filename="chart.png"))
    plt.close()

# --- Bot Events ---
@bot.event
async def on_ready():
    print(f"Logged in as {bot.user.name}")
    scan_markets.start()
    asyncio.create_task(listen_tradingview())

# --- Run the Bot ---
bot.run(DISCORD_TOKEN)
