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
import plotly.express as px

app = FastAPI()

# --- CONFIGURATION & DATABASE ---
URL_SB = os.getenv("URL_SB")
KEY_SB = os.getenv("KEY_SB")
try:
    supabase: Client = create_client(URL_SB, KEY_SB)
except:
    supabase = None

# --- V3.2 CLEAN & STABLE ASSET LIST (25 TICKERS) ---
TICKERS = {
    # Energy
    "Crude_Oil": "CL=F", "Brent_Oil": "BZ=F", "Natural_Gas": "NG=F", 
    "Gasoline": "RB=F", "Heating_Oil": "HO=F", "Uranium": "URA",
    # Industrial Metals
    "Copper": "HG=F", "Aluminum": "ALI=F", "Lithium": "LIT", "Steel": "SLX",
    # Precious Metals
    "Gold": "GC=F", "Silver": "SI=F", "Platinum": "PL=F", "Palladium": "PA=F",
    # Agriculture
    "Corn": "ZC=F", "Wheat": "ZW=F", "Soybeans": "ZS=F", "Coffee": "KC=F", 
    "Sugar": "SB=F", "Cocoa": "CC=F", "Cotton": "CT=F",
    # Macro Indicators
    "SP500": "^GSPC", "VIX_Index": "^VIX", "Bitcoin": "BTC-USD", "Carbon_Credits": "KRBN"
}

# --- QUANT & AI ENGINE ---

def calculate_rsi(prices, period=14):
    if len(prices) < period: return 50
    delta = pd.Series(prices).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return round(rsi.iloc[-1], 2)

def calculate_forecast(prices_list):
    y = np.array(prices_list)
    X = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, y)
    confidence = round(model.score(X, y), 4)
    prediction = round(model.predict(np.array([[len(y)]]))[0], 2)
    return prediction, confidence

# --- BACKGROUND WORKER V3.2 (No-Error Mode) ---
def background_worker():
    """Updates 25 rock-solid assets every 15 minutes."""
    while True:
        print(f"[{datetime.datetime.now()}] Refreshing 25 stable assets...")
        for name, ticker_id in TICKERS.items():
            try:
                ticker = yf.Ticker(ticker_id)
                hist = ticker.history(period="5d") 
                if hist.empty:
                    continue
                
                price = round(hist['Close'].iloc[-1], 2)
                mean_p = hist['Close'].mean()
                trend = "BULLISH" if price > mean_p else "BEARISH"
                
                if supabase:
                    supabase.table("precios_historicos").insert({
                        "activo": name, 
                        "precio": price, 
                        "tendencia": trend
                    }).execute()
                
                time.sleep(1) # Safety pause
            except Exception as e:
                print(f"❌ Worker skip on {name}: {e}")
        
        print("Update complete. Sleeping 900s...")
        time.sleep(900)

threading.Thread(target=background_worker, daemon=True).start()

# --- ENDPOINTS ---

@app.get("/")
def home():
    return {
        "platform": "Global Quant Radar V3.2",
        "status": "online",
        "stable_assets": len(TICKERS)
    }

@app.get("/visual-dashboard", response_class=HTMLResponse)
def get_dashboard():
    """Generates the clean terminal for 25 verified assets."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, 
        vertical_spacing=0.05, row_heights=[0.7, 0.3],
        subplot_titles=("Market Performance (%)", "RSI Momentum Indicators")
    )
    
    # Using a cleaner color palette
    colors = px.colors.qualitative.Alphabet + px.colors.qualitative.Dark24
    
    for i, (name, ticker_id) in enumerate(TICKERS.items()):
        try:
            hist = yf.Ticker(ticker_id).history(period="7d", interval="1h")
            if hist.empty or len(hist) < 2: continue
            
            perf = (hist['Close'] / hist['Close'].iloc[0] - 1) * 100
            color = colors[i % len(colors)]
            
            fig.add_trace(go.Scatter(x=hist.index, y=perf, name=name, 
                                     line=dict(color=color, width=1.5)), row=1, col=1)
            
            delta = hist['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = -delta.where(delta < 0, 0).rolling(14).mean()
            rsi_vals = 100 - (100 / (1 + (gain/loss + 1e-9)))
            fig.add_trace(go.Scatter(x=hist.index, y=rsi_vals, showlegend=False, 
                                     line=dict(color=color, width=1, dash='dot'), opacity=0.4), row=2, col=1)
        except: continue

    fig.update_layout(
        template="plotly_dark", height=900, 
        paper_bgcolor="#0a0a0a", plot_bgcolor="#0a0a0a",
        title_text="GLOBAL QUANT RADAR - STABLE V3.2", title_x=0.5,
        hovermode="x unified"
    )
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.3, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.3, row=2, col=1)

    return fig.to_html(full_html=True, include_plotlyjs='cdn')

# Standard API endpoints
@app.get("/market-intelligence")
def get_market_intelligence():
    analysis = {}
    for name, ticker_id in TICKERS.items():
        try:
            hist = yf.Ticker(ticker_id).history(period="30d", interval="1h")
            if hist.empty: continue
            prices = hist['Close']
            rsi = calculate_rsi(prices)
            analysis[name] = {
                "price": round(prices.iloc[-1], 2),
                "rsi_14": rsi,
                "sentiment": "OVERBOUGHT" if rsi > 70 else "OVERSOLD" if rsi < 30 else "NEUTRAL"
            }
        except: continue
    return analysis

@app.get("/historical-stats")
def get_historical_stats(asset: str = "Uranium"):
    if not supabase: return {"error": "DB connection failed"}
    try:
        response = supabase.table("precios_historicos").select("precio").eq("activo", asset).order("id", desc=True).limit(20).execute()
        prices = [row['precio'] for row in response.data]
        return {"asset": asset, "avg": round(sum(prices)/len(prices), 2), "points": len(prices)}
    except Exception as e: return {"error": str(e)}

@app.get("/premium-forecast")
def get_premium_forecast(asset: str = "Uranium"):
    if not supabase: return {"error": "DB connection failed"}
    try:
        response = supabase.table("precios_historicos").select("precio").eq("activo", asset).order("id", desc=True).limit(30).execute()
        prices = [row['precio'] for row in response.data]
        prices.reverse()
        if len(prices) < 10: return {"status": "collecting_data"}
        pred, conf = calculate_forecast(prices)
        return {"asset": asset, "forecast_24h": pred, "confidence": conf}
    except Exception as e: return {"error": str(e)}