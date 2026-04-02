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

# --- V3.0 EXPANDED ASSET LIST (34 TICKERS) ---
TICKERS = {
    # Energy
    "Crude_Oil": "CL=F", "Brent_Oil": "BZ=F", "Natural_Gas": "NG=F", 
    "Gasoline": "RB=F", "Heating_Oil": "HO=F", "Uranium": "URA",
    # Industrial Metals
    "Copper": "HG=F", "Aluminum": "ALI=F", "Nickel": "NICKEL", 
    "Zinc": "ZINC", "Lithium": "LIT", "Steel": "SLX", "Iron_Ore": "TIO=F",
    # Precious Metals
    "Gold": "GC=F", "Silver": "SI=F", "Platinum": "PL=F", "Palladium": "PA=F",
    # Agriculture
    "Corn": "ZC=F", "Wheat": "ZW=F", "Soybeans": "ZS=F", "Coffee": "KC=F", 
    "Sugar": "SB=F", "Cocoa": "CC=F", "Cotton": "CT=F", "Live_Cattle": "LC=F",
    # Macro & Future
    "Dollar_Index": "DX=F", "SP500": "^GSPC", "VIX_Index": "^VIX", 
    "Bitcoin": "BTC-USD", "Carbon_Credits": "KRBN"
}

# --- QUANT & AI ENGINE ---

def calculate_rsi(prices, period=14):
    """Calculates the Relative Strength Index."""
    if len(prices) < period: return 50
    delta = pd.Series(prices).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return round(rsi.iloc[-1], 2)

def calculate_forecast(prices_list):
    """Linear Regression to predict next trend value."""
    y = np.array(prices_list)
    X = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, y)
    confidence = round(model.score(X, y), 4)
    prediction = round(model.predict(np.array([[len(y)]]))[0], 2)
    return prediction, confidence

# --- BACKGROUND WORKER V3.0 ---
def background_worker():
    """Updates 34 assets every 15 minutes."""
    while True:
        print(f"[{datetime.datetime.now()}] Refreshing 34 global assets...")
        for name, ticker_id in TICKERS.items():
            try:
                ticker = yf.Ticker(ticker_id)
                hist = ticker.history(period="2d")
                if hist.empty: continue
                
                price = round(hist['Close'].iloc[-1], 2)
                mean_price = hist['Close'].mean()
                trend = "BULLISH" if price > mean_price else "BEARISH"
                
                if supabase:
                    supabase.table("precios_historicos").insert({
                        "activo": name, 
                        "precio": price, 
                        "tendencia": trend
                    }).execute()
                
                # Small pause to avoid Yahoo Finance rate limiting
                time.sleep(1.2) 
            except Exception as e:
                print(f"Worker Error on {name}: {e}")
        
        print("Global update complete. Sleeping for 900 seconds...")
        time.sleep(900)

# Start background thread
threading.Thread(target=background_worker, daemon=True).start()

# --- ENDPOINTS ---

@app.get("/")
def home():
    return {
        "platform": "Quant-Commodities Global Radar V3.0",
        "monitored_assets": len(TICKERS),
        "status": "online",
        "author": "Marcos"
    }

@app.get("/market-intelligence")
def get_market_intelligence():
    """Returns real-time quant metrics for all assets."""
    analysis = {}
    for name, ticker_id in TICKERS.items():
        try:
            hist = yf.Ticker(ticker_id).history(period="30d", interval="1h")
            if hist.empty: continue
            prices = hist['Close']
            current_val = round(prices.iloc[-1], 2)
            rsi = calculate_rsi(prices)
            sma20 = round(prices.rolling(20).mean().iloc[-1], 2)
            
            analysis[name] = {
                "price": current_val,
                "rsi_14": rsi,
                "sma_20": sma20,
                "sentiment": "OVERBOUGHT" if rsi > 70 else "OVERSOLD" if rsi < 30 else "NEUTRAL"
            }
        except: continue
    return analysis

@app.get("/visual-dashboard", response_class=HTMLResponse)
def get_dashboard():
    """Generates the institutional terminal for 34 assets."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, 
        vertical_spacing=0.05, row_heights=[0.7, 0.3],
        subplot_titles=("Global Cumulative Performance (%)", "RSI Momentum Indicators")
    )
    
    # Dynamic color palette for 34 assets
    colors = px.colors.qualitative.Alphabet + px.colors.qualitative.Dark24
    
    for i, (name, ticker_id) in enumerate(TICKERS.items()):
        try:
            hist = yf.Ticker(ticker_id).history(period="7d", interval="1h")
            if hist.empty: continue
            
            # Performance calculation
            perf = (hist['Close'] / hist['Close'].iloc[0] - 1) * 100
            color = colors[i % len(colors)]
            
            # Main Price Trace
            fig.add_trace(go.Scatter(
                x=hist.index, y=perf, name=name, 
                line=dict(color=color, width=1.5),
                hovertemplate='%{y:.2f}%'
            ), row=1, col=1)
            
            # RSI Trace
            delta = hist['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = -delta.where(delta < 0, 0).rolling(14).mean()
            rsi_vals = 100 - (100 / (1 + (gain/loss + 1e-9)))
            
            fig.add_trace(go.Scatter(
                x=hist.index, y=rsi_vals, showlegend=False, 
                line=dict(color=color, width=1, dash='dot'),
                opacity=0.4
            ), row=2, col=1)
        except: continue

    fig.update_layout(
        template="plotly_dark", height=950, 
        paper_bgcolor="#0a0a0a", plot_bgcolor="#0a0a0a",
        title_text="GLOBAL QUANT RADAR V3.0", title_x=0.5,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=9)),
        hovermode="x unified"
    )
    # Threshold lines for RSI
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.3, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.3, row=2, col=1)

    return fig.to_html(full_html=True, include_plotlyjs='cdn')

@app.get("/historical-stats")
def get_historical_stats(asset: str = "Uranium"):
    if not supabase: return {"error": "DB connection failed"}
    try:
        response = supabase.table("precios_historicos").select("precio").eq("activo", asset).order("id", desc=True).limit(20).execute()
        prices = [row['precio'] for row in response.data]
        if len(prices) < 2: return {"message": "Collecting more data points..."}
        return {
            "asset": asset, 
            "avg_price": round(sum(prices)/len(prices), 2), 
            "max": max(prices), 
            "min": min(prices),
            "data_points": len(prices)
        }
    except Exception as e: return {"error": str(e)}

@app.get("/premium-forecast")
def get_premium_forecast(asset: str = "Uranium"):
    if not supabase: return {"error": "DB connection failed"}
    try:
        response = supabase.table("precios_historicos").select("precio").eq("activo", asset).order("id", desc=True).limit(30).execute()
        prices = [row['precio'] for row in response.data]
        prices.reverse()
        if len(prices) < 15: return {"status": "warming_up", "current_points": len(prices)}
        prediction, confidence = calculate_forecast(prices)
        return {
            "asset": asset, 
            "forecast_24h": prediction, 
            "confidence_r2": confidence,
            "signal": "BUY/UP" if prediction > prices[-1] else "SELL/DOWN"
        }
    except Exception as e: return {"error": str(e)}