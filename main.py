from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import yfinance as yf
import pandas as pd
from supabase import create_client, Client
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
        for name, ticker_id in ALL_TICKERS.items():
            try:
                ticker = yf.Ticker(ticker_id)
                hist = ticker.history(period="5d") 
                if hist.empty: continue
                price = round(hist['Close'].iloc[-1], 2)
                if supabase:
                    supabase.table("precios_historicos").insert({"activo": name, "precio": price}).execute()
                time.sleep(1)
            except: continue
        time.sleep(900)

threading.Thread(target=background_worker, daemon=True).start()

# --- DASHBOARD GENERATOR ---
@app.get("/visual-dashboard", response_class=HTMLResponse)
def get_dashboard():
    market_data = []
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, 
        vertical_spacing=0.12, 
        row_heights=[0.65, 0.35],
        subplot_titles=("RELATIVE PERFORMANCE (%)", "MOMENTUM INDEX (RSI 14)")
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
                
                start_p = hist['Close'].iloc[0]
                current_p = hist['Close'].iloc[-1]
                perf = ((current_p / start_p) - 1) * 100
                rsi = calculate_rsi(hist['Close'])
                
                market_data.append({"Asset": name, "Sector": sector, "Price": round(current_p, 2), "Perf": round(perf, 2), "RSI": rsi})
                color = sector_colors[sector]
                
                # Perf Trace
                fig.add_trace(go.Scatter(
                    x=hist.index, y=((hist['Close']/start_p)-1)*100, 
                    name=name, legendgroup=sector,
                    line=dict(color=color, width=2),
                    hovertemplate='<b>'+name+'</b>: %{y:.2f}%<extra></extra>'
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

    # --- UI FIX: BUTTONS & VISIBILITY ---
    buttons = [dict(method="restyle", label="GLOBAL VIEW", args=[{"visible": [True] * trace_counter}])]
    for target_sector in CATEGORIZED_TICKERS.keys():
        visibility = []
        for sector, assets in CATEGORIZED_TICKERS.items():
            for _ in assets:
                val = (sector == target_sector)
                visibility.extend([val, val])
        buttons.append(dict(method="restyle", label=target_sector.upper(), args=[{"visible": visibility}]))

    fig.update_layout(
        template="plotly_dark", height=900, 
        margin=dict(t=180, b=50, l=60, r=60), 
        paper_bgcolor="#0a0a0a", plot_bgcolor="#0a0a0a",
        title_text="GLOBAL QUANT TERMINAL V3.8", title_x=0.5, title_y=0.98,
        hovermode="x unified",
        legend=dict(
            itemclick="toggle", # One click: hide/show
            itemdoubleclick="isolate", # Double click: see ONLY this one
            font=dict(size=10), orientation="v", x=1.02, y=0.5
        ),
        updatemenus=[dict(
            type="buttons", direction="right", x=0.5, y=1.2, 
            xanchor="center", yanchor="top",
            buttons=buttons,
            bgcolor="#111827", # Darker background
            font=dict(color="#ffffff", size=11), # Pure white text
            active=0, bordercolor="#374151"
        )]
    )

    # --- UI FIX: TABLE ALIGNMENT ---
    df_market = pd.DataFrame(market_data).sort_values(by="Perf", ascending=False)
    table_html = """
    <div style="background-color: #0a0a0a; color: white; font-family: Arial, sans-serif; padding: 40px;">
        <h2 style="text-align: center; color: #64748b; letter-spacing: 2px;">MARKET SUMMARY</h2>
        <table style="width: 100%; border-collapse: collapse; margin-top: 20px; table-layout: fixed;">
            <thead>
                <tr style="background-color: #111827; color: #94a3b8; text-align: left; border-bottom: 2px solid #1f2937;">
                    <th style="padding: 15px; width: 25%;">ASSET</th>
                    <th style="padding: 15px; width: 20%;">SECTOR</th>
                    <th style="padding: 15px; width: 20%;">PRICE</th>
                    <th style="padding: 15px; width: 20%;">7D PERF</th>
                    <th style="padding: 15px; width: 15%;">RSI</th>
                </tr>
            </thead>
            <tbody>
    """
    for _, row in df_market.iterrows():
        p_color = "#10b981" if row['Perf'] > 0 else "#ef4444"
        table_html += f"""
                <tr style="border-bottom: 1px solid #1f2937; text-align: left;">
                    <td style="padding: 12px; font-weight: bold;">{row['Asset']}</td>
                    <td style="padding: 12px; color: #4b5563;">{row['Sector']}</td>
                    <td style="padding: 12px;">{row['Price']}</td>
                    <td style="padding: 12px; color: {p_color}; font-weight: bold;">{row['Perf']}%</td>
                    <td style="padding: 12px;">{row['RSI']}</td>
                </tr>
        """
    table_html += "</tbody></table></div>"

    chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
    return HTMLResponse(content=f"<html><body style='margin:0; background:#0a0a0a;'>{chart_html}{table_html}</body></html>")