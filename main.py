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

# --- CATEGORIZED ASSET LIST ---
CATEGORIZED_TICKERS = {
    "Energy": {
        "Crude_Oil": "CL=F", "Brent_Oil": "BZ=F", "Natural_Gas": "NG=F", 
        "Gasoline": "RB=F", "Heating_Oil": "HO=F", "Uranium": "URA"
    },
    "Metals": {
        "Copper": "HG=F", "Aluminum": "ALI=F", "Lithium": "LIT", 
        "Steel": "SLX", "Gold": "GC=F", "Silver": "SI=F", 
        "Platinum": "PL=F", "Palladium": "PA=F"
    },
    "Agriculture": {
        "Corn": "ZC=F", "Wheat": "ZW=F", "Soybeans": "ZS=F", 
        "Coffee": "KC=F", "Sugar": "SB=F", "Cocoa": "CC=F", "Cotton": "CT=F"
    },
    "Macro_Crypto": {
        "SP500": "^GSPC", "VIX_Index": "^VIX", "Bitcoin": "BTC-USD", "Carbon_Credits": "KRBN"
    }
}

ALL_TICKERS = {k: v for cat in CATEGORIZED_TICKERS.values() for k, v in cat.items()}

# --- QUANT ENGINE ---

def calculate_rsi(prices, period=14):
    if len(prices) < period: return 50
    delta = pd.Series(prices).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return round(rsi.iloc[-1], 2)

# --- BACKGROUND WORKER ---
def background_worker():
    while True:
        print(f"[{datetime.datetime.now()}] Worker: Syncing 25 assets...")
        for name, ticker_id in ALL_TICKERS.items():
            try:
                ticker = yf.Ticker(ticker_id)
                hist = ticker.history(period="5d") 
                if hist.empty: continue
                price = round(hist['Close'].iloc[-1], 2)
                if supabase:
                    supabase.table("precios_historicos").insert({
                        "activo": name, "precio": price, "tendencia": "N/A"
                    }).execute()
                time.sleep(1)
            except Exception as e:
                print(f"Error on {name}: {e}")
        time.sleep(900)

threading.Thread(target=background_worker, daemon=True).start()

# --- DASHBOARD GENERATOR ---

@app.get("/visual-dashboard", response_class=HTMLResponse)
def get_dashboard():
    # 1. Gather Data and Compute Stats for Table
    market_data = []
    
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, 
        vertical_spacing=0.07, row_heights=[0.7, 0.3],
        subplot_titles=("Relative Sector Performance (%)", "RSI Momentum Indicators")
    )
    
    sector_colors = {
        "Energy": "#ef4444", "Metals": "#f59e0b", 
        "Agriculture": "#10b981", "Macro_Crypto": "#3b82f6"
    }

    trace_counter = 0
    for sector, assets in CATEGORIZED_TICKERS.items():
        for name, ticker_id in assets.items():
            try:
                hist = yf.Ticker(ticker_id).history(period="7d", interval="1h")
                if hist.empty or len(hist) < 2: continue
                
                # Performance & Stats
                current_price = hist['Close'].iloc[-1]
                start_price = hist['Close'].iloc[0]
                daily_perf = ((current_price / start_price) - 1) * 100
                rsi = calculate_rsi(hist['Close'])
                
                market_data.append({
                    "Asset": name, "Sector": sector, 
                    "Price": round(current_price, 2), 
                    "Perf %": round(daily_perf, 2),
                    "RSI": rsi
                })

                color = sector_colors[sector]
                
                # Performance Trace
                fig.add_trace(go.Scatter(
                    x=hist.index, y=(hist['Close']/start_price-1)*100, 
                    name=name, legendgroup=sector,
                    line=dict(color=color, width=1.8),
                    hovertemplate='%{y:.2f}%'
                ), row=1, col=1)
                
                # RSI Trace
                delta = hist['Close'].diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean()
                loss = -delta.where(delta < 0, 0).rolling(14).mean()
                rsi_vals = 100 - (100 / (1 + (gain/loss + 1e-9)))
                fig.add_trace(go.Scatter(
                    x=hist.index, y=rsi_vals, showlegend=False,
                    legendgroup=sector, line=dict(color=color, width=1, dash='dot'),
                    opacity=0.3
                ), row=2, col=1)
                
                trace_counter += 2
            except: continue

    # 2. Build Buttons
    buttons = [dict(method="restyle", label="View All", args=[{"visible": [True] * trace_counter}])]
    for target_sector in CATEGORIZED_TICKERS.keys():
        visibility = []
        for sector, assets in CATEGORIZED_TICKERS.items():
            for _ in assets:
                val = (sector == target_sector)
                visibility.extend([val, val])
        buttons.append(dict(method="restyle", label=target_sector, args=[{"visible": visibility}]))

    fig.update_layout(
        template="plotly_dark", height=800, 
        paper_bgcolor="#0a0a0a", plot_bgcolor="#0a0a0a",
        title_text="GLOBAL QUANT RADAR V3.6", title_x=0.5,
        hovermode="x unified",
        updatemenus=[dict(
            type="buttons", direction="right", x=0.5, y=1.08, 
            xanchor="center", buttons=buttons, bgcolor="#1e293b"
        )]
    )

    # 3. Create HTML Summary Table (The "Perfection" Step)
    df_market = pd.DataFrame(market_data).sort_values(by="Perf %", ascending=False)
    
    def get_color(val):
        return "#10b981" if val > 0 else "#ef4444"

    table_html = """
    <div style="background-color: #0a0a0a; color: white; font-family: Arial; padding: 20px;">
        <h2 style="text-align: center;">Market Intelligence Summary</h2>
        <table style="width: 100%; border-collapse: collapse; text-align: left; background-color: #111;">
            <thead>
                <tr style="border-bottom: 2px solid #333;">
                    <th style="padding: 12px;">Asset</th>
                    <th style="padding: 12px;">Sector</th>
                    <th style="padding: 12px;">Price</th>
                    <th style="padding: 12px;">7d Perf %</th>
                    <th style="padding: 12px;">RSI (14)</th>
                </tr>
            </thead>
            <tbody>
    """
    for _, row in df_market.iterrows():
        color = get_color(row['Perf %'])
        table_html += f"""
                <tr style="border-bottom: 1px solid #222;">
                    <td style="padding: 10px;">{row['Asset']}</td>
                    <td style="padding: 10px; font-size: 0.8em; color: #888;">{row['Sector']}</td>
                    <td style="padding: 10px; font-weight: bold;">{row['Price']}</td>
                    <td style="padding: 10px; color: {color};">{row['Perf %']}%</td>
                    <td style="padding: 10px;">{row['RSI']}</td>
                </tr>
        """
    table_html += "</tbody></table></div>"

    # Combine Plot + Table
    full_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
    final_page = f"""
    <html>
        <head><title>Quant Terminal V3.6</title></head>
        <body style="background-color: #0a0a0a; margin: 0;">
            {full_html}
            {table_html}
        </body>
    </html>
    """
    return HTMLResponse(content=final_page)

# ... (Keep other endpoints like stats/forecast as they are) ...
@app.get("/")
def home():
    return {"status": "V3.6 Active", "language": "English/Spanish Hybrid"}