from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import yfinance as yf
import pandas as pd
from supabase import create_client, Client
from sklearn.linear_model import LinearRegression
import numpy as np
import os
import datetime
import threading
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots

app = FastAPI()

# --- CONFIGURATION & DATABASE ---
URL_SB = os.getenv("URL_SB")
KEY_SB = os.getenv("KEY_SB")
try:
    supabase: Client = create_client(URL_SB, KEY_SB)
except:
    supabase = None

# The 7 Strategic Assets
TICKERS = {
    "Gold": "GC=F", 
    "Oil": "CL=F", 
    "Copper": "HG=F", 
    "Silver": "SI=F", 
    "Natural_Gas": "NG=F", 
    "Lithium": "LIT", 
    "Uranium": "URA" 
}

# --- QUANT & AI ENGINE ---

def calculate_rsi(prices, period=14):
    """Calculates the Relative Strength Index."""
    if len(prices) < period: return 50
    delta = pd.Series(prices).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return round(rsi.iloc[-1], 2)

def calculate_forecast(prices_list):
    """Linear Regression to predict the next trend value."""
    y = np.array(prices_list)
    X = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, y)
    confidence = round(model.score(X, y), 4)
    next_index = np.array([[len(y)]])
    prediction = round(model.predict(next_index)[0], 2)
    return prediction, confidence

# --- BACKGROUND WORKER (The "Tap" that never stops) ---
def background_worker():
    """Fetches data every 15 minutes and saves to Supabase."""
    while True:
        print(f"[{datetime.datetime.now()}] Refreshing 15-min data for 7 assets...")
        for name, ticker_id in TICKERS.items():
            try:
                ticker = yf.Ticker(ticker_id)
                hist = ticker.history(period="2d")
                if len(hist) < 1: continue
                
                price = round(hist['Close'].iloc[-1], 2)
                # Trend based on the mean of the last 2 days
                trend = "BULLISH" if price > hist['Close'].mean() else "BEARISH"
                
                if supabase:
                    supabase.table("precios_historicos").insert({
                        "activo": name, 
                        "precio": price, 
                        "tendencia": trend
                    }).execute()
            except Exception as e:
                print(f"Worker Error on {name}: {e}")
                continue
        
        print("Update complete. Sleeping for 900 seconds...")
        time.sleep(900)

# Start the worker in a separate thread
threading.Thread(target=background_worker, daemon=True).start()

# --- ENDPOINTS ---

@app.get("/")
def home():
    return {
        "platform": "Quant-Commodities AI V2.0",
        "status": "online",
        "endpoints": ["/market-intelligence", "/historical-stats", "/premium-forecast", "/visual-dashboard"]
    }

# 1. Market Intelligence (All 7 Assets)
@app.get("/market-intelligence")
def get_market_intelligence():
    """Returns price, RSI, SMA and Sentiment for all assets."""
    analysis = {}
    for name, ticker_id in TICKERS.items():
        try:
            ticker = yf.Ticker(ticker_id)
            hist = ticker.history(period="30d", interval="1h")
            prices = hist['Close']
            current_val = round(prices.iloc[-1], 2)
            
            rsi = calculate_rsi(prices)
            sma20 = round(prices.rolling(20).mean().iloc[-1], 2)
            
            analysis[name] = {
                "price": current_val,
                "rsi_14": rsi,
                "sma_20": sma20,
                "trend": "BULLISH" if current_val > sma20 else "BEARISH",
                "sentiment": "OVERBOUGHT" if rsi > 70 else "OVERSOLD" if rsi < 30 else "NEUTRAL"
            }
        except: continue
    return analysis

# 2. Historical Stats (Specific Asset)
@app.get("/historical-stats")
def get_historical_stats(asset: str = "Uranium"):
    """Fetches historical records from Supabase and returns stats."""
    if not supabase: return {"error": "DB connection failed"}
    if asset not in TICKERS: return {"error": "Invalid asset"}

    try:
        response = supabase.table("precios_historicos").select("precio").eq("activo", asset).order("id", desc=True).limit(20).execute()
        prices = [row['precio'] for row in response.data]
        
        if len(prices) < 2: return {"message": "Collecting more data..."}
        
        return {
            "asset": asset,
            "avg_price": round(sum(prices)/len(prices), 2),
            "volatility": round(pd.Series(prices).std(), 4),
            "max": max(prices),
            "min": min(prices)
        }
    except Exception as e: return {"error": str(e)}

# 3. Premium AI Forecast (Specific Asset)
@app.get("/premium-forecast")
def get_premium_forecast(asset: str = "Uranium"):
    """Advanced AI Trend Prediction."""
    if not supabase: return {"error": "DB connection failed"}
    if asset not in TICKERS: return {"error": "Invalid asset"}

    try:
        response = supabase.table("precios_historicos").select("precio").eq("activo", asset).order("id", desc=True).limit(30).execute()
        prices = [row['precio'] for row in response.data]
        prices.reverse()
        
        if len(prices) < 15: return {"status": "warming_up", "data_points": len(prices)}
        
        prediction, confidence = calculate_forecast(prices)
        current = prices[-1]
        rsi = calculate_rsi(prices)
        
        return {
            "asset": asset,
            "current_price": current,
            "rsi": rsi,
            "forecast_next_24h": prediction,
            "ai_confidence": confidence,
            "signal": "UPWARD" if prediction > current else "DOWNWARD",
            "alert": "High reversal risk" if (rsi > 70 or rsi < 30) else "Stable trend"
        }
    except Exception as e: return {"error": str(e)}

# 4. Visual Dashboard (Institutional Chart)
@app.get("/visual-dashboard", response_class=HTMLResponse)
def get_dashboard():
    """Generates the professional dark chart for all 7 assets."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, 
        vertical_spacing=0.07, row_heights=[0.7, 0.3],
        subplot_titles=("Cumulative Performance (%)", "RSI Momentum (14)")
    )
    
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692']
    
    for i, (name, ticker_id) in enumerate(TICKERS.items()):
        hist = yf.Ticker(ticker_id).history(period="7d", interval="1h")
        if hist.empty: continue
        
        # Perf %
        perf = (hist['Close'] / hist['Close'].iloc[0] - 1) * 100
        fig.add_trace(go.Scatter(x=hist.index, y=perf, name=name, line=dict(color=colors[i], width=2)), row=1, col=1)
        
        # RSI
        delta = hist['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rsi_vals = 100 - (100 / (1 + (gain/loss)))
        fig.add_trace(go.Scatter(x=hist.index, y=rsi_vals, showlegend=False, line=dict(color=colors[i], width=1)), row=2, col=1)

    fig.update_layout(
        template="plotly_dark", height=850, paper_bgcolor="#121212", plot_bgcolor="#1d1d1d",
        title_text="Quant-Commodities Institutional Terminal", title_x=0.5,
        hovermode="x unified", font=dict(family="Arial", size=12)
    )
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.3, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.3, row=2, col=1)

    return fig.to_html(full_html=True, include_plotlyjs='cdn')