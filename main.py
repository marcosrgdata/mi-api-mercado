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
    delta = prices.diff()
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
        # 1. FETCH DATA (BULK)
        ticker_ids = list(ALL_TICKERS.values())
        raw_data = yf.download(ticker_ids, period="7d", interval="1h", group_by='ticker', threads=True)
        
        market_data = []
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, 
            vertical_spacing=0.12, row_heights=[0.65, 0.35],
            subplot_titles=("RELATIVE PERFORMANCE (%)", "MOMENTUM INDEX (RSI 14)")
        )
        
        sector_colors = {"Energy": "#ef4444", "Metals": "#f59e0b", "Agriculture": "#10b981", "Macro_Crypto": "#3b82f6"}
        trace_counter = 0

        for sector, assets in CATEGORIZED_TICKERS.items():
            for name, tid in assets.items():
                try:
                    hist = raw_data[tid].dropna() if len(ticker_ids) > 1 else raw_data.dropna()
                    if hist.empty or len(hist) < 2: continue
                    
                    start_p = hist['Close'].iloc[0]
                    current_p = hist['Close'].iloc[-1]
                    perf = ((current_p / start_p) - 1) * 100
                    rsi = calculate_rsi(hist['Close'])
                    
                    market_data.append({"Asset": name, "Sector": sector, "Price": round(current_p, 2), "Perf": round(perf, 2), "RSI": rsi})
                    color = sector_colors[sector]
                    
                    # PERFORMANCE TRACE
                    fig.add_trace(go.Scatter(
                        x=hist.index, y=((hist['Close']/start_p)-1)*100, 
                        name=name, 
                        legendgroup=name, # GROUP BY ASSET NAME FOR INDIVIDUAL ISOLATION
                        line=dict(color=color, width=2),
                        hovertemplate='<b>'+name+'</b>: %{y:.2f}%<extra></extra>'
                    ), row=1, col=1)
                    
                    # RSI TRACE
                    delta = hist['Close'].diff()
                    gain = delta.where(delta > 0, 0).rolling(14).mean()
                    loss = -delta.where(delta < 0, 0).rolling(14).mean()
                    rsi_line = 100 - (100 / (1 + (gain/(loss + 1e-9))))
                    
                    fig.add_trace(go.Scatter(
                        x=hist.index, y=rsi_line, 
                        showlegend=False,
                        legendgroup=name, # LINKED TO PRICE TRACE
                        line=dict(color=color, width=1, dash='dot'),
                        opacity=0.3
                    ), row=2, col=1)
                    trace_counter += 2
                except: continue

        # 2. BUTTONS CONFIG (SECTOR FILTERING)
        buttons = [dict(method="restyle", label="GLOBAL VIEW", args=[{"visible": [True] * trace_counter}])]
        for target_sector in CATEGORIZED_TICKERS.keys():
            visibility = []
            for sector, assets in CATEGORIZED_TICKERS.items():
                for _ in assets:
                    val = (sector == target_sector)
                    visibility.extend([val, val])
            buttons.append(dict(method="restyle", label=target_sector.upper(), args=[{"visible": visibility}]))

        fig.update_layout(
            template="plotly_dark", height=850, margin=dict(t=180, b=50, l=60, r=60),
            paper_bgcolor="#0a0a0a", plot_bgcolor="#0a0a0a",
            title_text="GLOBAL QUANT TERMINAL V3.14", title_x=0.5, title_y=0.98,
            hovermode="x unified",
            legend=dict(
                itemclick="toggleothers", # ONE CLICK ISOLATES ASSET (PRICE + RSI)
                itemdoubleclick="toggle", # DOUBLE CLICK BRINGS ALL BACK
                font=dict(size=10, color="white"), 
                orientation="v", x=1.02, y=0.5
            ),
            updatemenus=[dict(
                type="buttons", direction="right", x=0.5, y=1.22, xanchor="center",
                buttons=buttons, bgcolor="#1e293b", font=dict(color="#ffffff", size=11),
                active=-1, bordercolor="#334155"
            )]
        )

        # 3. TABLE & CUSTOM CSS
        df_market = pd.DataFrame(market_data).sort_values(by="Perf", ascending=False)
        table_rows = "".join([f"""
            <tr>
                <td style="padding: 12px; font-weight: bold; text-align: left;">{r['Asset']}</td>
                <td style="padding: 12px; color: #4b5563; text-align: left;">{r['Sector']}</td>
                <td style="padding: 12px; font-family: monospace; text-align: left;">{r['Price']}</td>
                <td style="padding: 12px; color: {'#10b981' if r['Perf'] > 0 else '#ef4444'}; font-weight: bold; text-align: left;">{r['Perf']}%</td>
                <td style="padding: 12px; text-align: left;">{r['RSI']}</td>
            </tr>""" for _, r in df_market.iterrows()])

        custom_css = """
        <style>
            rect.updatemenu-item-rect:hover { fill: #334155 !important; }
            rect.updatemenu-item-rect.active { fill: #3b82f6 !important; }
            text.updatemenu-item-text { fill: #ffffff !important; font-weight: bold !important; }
            table { width: 100%; border-collapse: collapse; table-layout: fixed; margin-top: 20px; color: white; }
            th { padding: 15px; text-align: left; background-color: #111827; color: #94a3b8; border-bottom: 2px solid #1f2937; }
            tr { border-bottom: 1px solid #1f2937; }
        </style>
        """

        table_html = f"""
        <div style="background-color: #0a0a0a; padding: 40px; font-family: sans-serif;">
            <h2 style="text-align: center; color: #64748b; letter-spacing: 2px;">MARKET INTELLIGENCE SUMMARY</h2>
            <table>
                <thead>
                    <tr><th>ASSET</th><th>SECTOR</th><th>PRICE</th><th>7D PERF</th><th>RSI</th></tr>
                </thead>
                <tbody>{table_rows}</tbody>
            </table>
        </div>"""
        
        return HTMLResponse(content=f"<html><head>{custom_css}</head><body style='margin:0; background:#0a0a0a;'>{fig.to_html(full_html=False, include_plotlyjs='cdn')}{table_html}</body></html>")
    
    except Exception as e:
        return HTMLResponse(content=f"<html><body style='background:#111; color:white; padding:20px;'><h1>Dashboard Error</h1><code>{str(e)}</code></body></html>", status_code=500)