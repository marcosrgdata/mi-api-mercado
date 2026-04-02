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

# --- QUANT & PREDICTION ENGINE ---
def get_projection(data_series, hours_ahead=24):
    """Calculates a linear projection for the next N hours."""
    y = data_series.values.reshape(-1, 1)
    X = np.arange(len(y)).reshape(-1, 1)
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict future points
    future_X = np.arange(len(y), len(y) + hours_ahead).reshape(-1, 1)
    future_y = model.predict(future_X)
    
    return future_y.flatten()

# --- BACKGROUND WORKER ---
def background_worker():
    while True:
        for name, ticker_id in ALL_TICKERS.items():
            try:
                t = yf.Ticker(ticker_id)
                h = t.history(period="1d")
                if not h.empty and supabase:
                    p = round(h['Close'].iloc[-1], 2)
                    supabase.table("precios_historicos").insert({"activo": name, "precio": p}).execute()
                time.sleep(1.5)
            except: continue
        time.sleep(900)

threading.Thread(target=background_worker, daemon=True).start()

# --- DASHBOARD GENERATOR ---
@app.get("/visual-dashboard", response_class=HTMLResponse)
def get_dashboard():
    try:
        # 1. BULK DATA FETCH
        ticker_ids = list(ALL_TICKERS.values())
        raw_data = yf.download(ticker_ids, period="7d", interval="1h", group_by='ticker', threads=True)
        
        market_data = []
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, 
            vertical_spacing=0.12, row_heights=[0.65, 0.35],
            subplot_titles=("RELATIVE PERFORMANCE & AI PROJECTION", "MOMENTUM INDEX (RSI 14)")
        )
        
        sector_colors = {"Energy": "#ef4444", "Metals": "#f59e0b", "Agriculture": "#10b981", "Macro_Crypto": "#3b82f6"}
        trace_counter = 0

        for sector, assets in CATEGORIZED_TICKERS.items():
            for name, tid in assets.items():
                try:
                    hist = raw_data[tid].dropna() if len(ticker_ids) > 1 else raw_data.dropna()
                    if hist.empty or len(hist) < 10: continue
                    
                    start_p = hist['Close'].iloc[0]
                    current_p = hist['Close'].iloc[-1]
                    perf_series = ((hist['Close'] / start_p) - 1) * 100
                    color = sector_colors[sector]
                    
                    # --- A. HISTORICAL TRACE ---
                    fig.add_trace(go.Scatter(
                        x=hist.index, y=perf_series, 
                        name=name, legendgroup=name,
                        line=dict(color=color, width=2.5),
                        hovertemplate='<b>'+name+'</b>: %{y:.2f}%<extra></extra>'
                    ), row=1, col=1)
                    
                    # --- B. AI PROJECTION TRACE (The new faint line) ---
                    projection_y = get_projection(perf_series, hours_ahead=24)
                    # Create future timestamps starting from the last known date
                    future_index = pd.date_range(start=hist.index[-1], periods=25, freq='H')[1:]
                    
                    # We prepend the last real value to the projection for visual continuity
                    proj_x = [hist.index[-1]] + list(future_index)
                    proj_y = [perf_series.iloc[-1]] + list(projection_y)
                    
                    fig.add_trace(go.Scatter(
                        x=proj_x, y=proj_y,
                        name=f"{name} Forecast",
                        legendgroup=name, showlegend=False,
                        line=dict(color=color, width=2, dash='dot'),
                        opacity=0.35, # Faint line
                        hovertemplate='<b>'+name+' Forecast</b>: %{y:.2f}%<extra></extra>'
                    ), row=1, col=1)

                    # --- C. RSI TRACE ---
                    delta = hist['Close'].diff()
                    gain = delta.where(delta > 0, 0).rolling(14).mean()
                    loss = -delta.where(delta < 0, 0).rolling(14).mean()
                    rsi_line = 100 - (100 / (1 + (gain/(loss + 1e-9))))
                    
                    fig.add_trace(go.Scatter(
                        x=hist.index, y=rsi_line, showlegend=False,
                        legendgroup=name, line=dict(color=color, width=1, dash='dot'), opacity=0.2
                    ), row=2, col=1)
                    
                    market_data.append({"Asset": name, "Sector": sector, "Price": round(current_p, 2), "Perf": round(perf_series.iloc[-1], 2)})
                    trace_counter += 3 # Three traces: Price, Forecast, RSI
                except: continue

        # 2. BUTTONS & UI
        buttons = [dict(method="restyle", label="GLOBAL VIEW", args=[{"visible": [True] * trace_counter}])]
        for target_sector in CATEGORIZED_TICKERS.keys():
            visibility = []
            for sector, assets in CATEGORIZED_TICKERS.items():
                for _ in assets:
                    val = (sector == target_sector)
                    visibility.extend([val, val, val]) # Triplet for Price, Forecast, RSI
            buttons.append(dict(method="restyle", label=target_sector.upper(), args=[{"visible": visibility}]))

        fig.update_layout(
            template="plotly_dark", height=850, margin=dict(t=180, b=50, l=60, r=60),
            paper_bgcolor="#0a0a0a", plot_bgcolor="#0a0a0a",
            title_text="GLOBAL QUANT PREDICTIVE TERMINAL V3.17", title_x=0.5, title_y=0.98,
            legend=dict(itemclick="toggleothers", itemdoubleclick="toggle", font=dict(size=10), orientation="v", x=1.02, y=0.5),
            updatemenus=[dict(
                type="buttons", direction="right", x=0.5, y=1.22, xanchor="center",
                buttons=buttons, bgcolor="#1e293b", font=dict(color="#ffffff", size=11), bordercolor="#334155"
            )]
        )

        # 3. CSS & TABLE
        custom_css = """
        <style>
            rect.updatemenu-item-rect { fill: #1e293b !important; }
            rect.updatemenu-item-rect:hover { fill: #334155 !important; }
            rect.updatemenu-item-rect[fill="#F4F4F4"], rect.updatemenu-item-rect.active { fill: #2563eb !important; }
            text.updatemenu-item-text { fill: #ffffff !important; font-weight: bold !important; pointer-events: none !important; }
            table { width: 100%; border-collapse: collapse; margin-top: 20px; color: white; }
            th { padding: 15px; text-align: left; background-color: #111827; color: #94a3b8; border-bottom: 2px solid #1f2937; }
            tr { border-bottom: 1px solid #1f2937; }
        </style>
        """
        table_html = f"<div style='background:#0a0a0a; padding:40px; font-family:sans-serif;'><h2 style='text-align:center; color:#64748b;'>MARKET SUMMARY</h2><table><thead><tr><th>ASSET</th><th>SECTOR</th><th>PRICE</th><th>7D PERF</th></tr></thead>"
        for r in market_data:
            c = "#10b981" if r['Perf'] > 0 else "#ef4444"
            table_html += f"<tr><td><b>{r['Asset']}</b></td><td style='color:#4b5563;'>{r['Sector']}</td><td>{r['Price']}</td><td style='color:{c};'>{r['Perf']}%</td></tr>"
        table_html += "</table></div>"
        
        return HTMLResponse(content=f"<html><head>{custom_css}</head><body style='margin:0; background:#0a0a0a;'>{fig.to_html(full_html=False, include_plotlyjs='cdn')}{table_html}</body></html>")
    
    except Exception as e:
        return HTMLResponse(content=f"<html><body style='background:#111; color:white; padding:20px;'><h1>Dashboard Error</h1><code>{str(e)}</code></body></html>", status_code=500)