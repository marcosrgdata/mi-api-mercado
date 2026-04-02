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

# --- V3.1 FIXED ASSET LIST (Verified Tickers) ---
TICKERS = {
    # Energy
    "Crude_Oil": "CL=F", "Brent_Oil": "BZ=F", "Natural_Gas": "NG=F", 
    "Gasoline": "RB=F", "Heating_Oil": "HO=F", "Uranium": "URA",
    # Industrial Metals (Fixed Tickers)
    "Copper": "HG=F", "Aluminum": "ALI=F", "Nickel": "JJN", 
    "Zinc": "ZNC=F", "Lithium": "LIT", "Steel": "SLX", "Iron_Ore": "PICK",
    # Precious Metals
    "Gold": "GC=F", "Silver": "SI=F", "Platinum": "PL=F", "Palladium": "PA=F",
    # Agriculture
    "Corn": "ZC=F", "Wheat": "ZW=F", "Soybeans": "ZS=F", "Coffee": "KC=F", 
    "Sugar": "SB=F", "Cocoa": "CC=F", "Cotton": "CT=F", "Live_Cattle": "LE=F",
    # Macro & Future (Fixed DXY)
    "Dollar_Index": "DX-Y.NYB", "SP500": "^GSPC", "VIX_Index": "^VIX", 
    "Bitcoin": "BTC-USD", "Carbon_Credits": "KRBN"
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

# --- BACKGROUND WORKER V3.1 (Robust Mode) ---
def background_worker():
    while True:
        print(f"[{datetime.datetime.now()}] Refreshing 34 global assets...")
        for name, ticker_id in TICKERS.items():
            try:
                ticker = yf.Ticker(ticker_id)
                hist = ticker.history(period="5d") # Aumentado a 5d para mayor seguridad
                if hist.empty:
                    print(f"⚠️ Warning: No data for {name} ({ticker_id})")
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
                
                time.sleep(1.5) # Respiro para la API
            except Exception as e:
                print(f"❌ Error on {name}: {e}")
        
        print("Update complete. Sleeping 900s...")
        time.sleep(900)

threading.Thread(target=background_worker, daemon=True).start()

# --- ENDPOINTS ---

@app.get("/")
def home():
    return {"status": "V3.1 Fixed", "assets": len(TICKERS)}

@app.get("/visual-dashboard", response_class=HTMLResponse)
def get_dashboard():
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, 
        vertical_spacing=0.05, row_heights=[0.7, 0.3],
        subplot_titles=("Performance (%)", "RSI Momentum")
    )
    
    colors = px.colors.qualitative.Alphabet + px.colors.qualitative.Dark24
    
    for i, (name, ticker_id) in enumerate(TICKERS.items()):
        try:
            # Pedimos un poco más de margen para evitar huecos (7d)
            hist = yf.Ticker(ticker_id).history(period="7d", interval="1h")
            if hist.empty or len(hist) < 2: continue
            
            perf = (hist['Close'] / hist['Close'].iloc[0] - 1) * 100
            color = colors[i % len(colors)]
            
            fig.add_trace(go.Scatter(x=hist.index, y=perf, name=name, line=dict(color=color, width=1.5)), row=1, col=1)
            
            delta = hist['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = -delta.where(delta < 0, 0).rolling(14).mean()
            rsi_vals = 100 - (100 / (1 + (gain/loss + 1e-9)))
            fig.add_trace(go.Scatter(x=hist.index, y=rsi_vals, showlegend=False, line=dict(color=color, width=1, dash='dot'), opacity=0.4), row=2, col=1)
        except: continue

    fig.update_layout(template="plotly_dark", height=900, paper_bgcolor="#0a0a0a", plot_bgcolor="#0a0a0a", title_text="GLOBAL RADAR V3.1", title_x=0.5)
    return fig.to_html(full_html=True, include_plotlyjs='cdn')