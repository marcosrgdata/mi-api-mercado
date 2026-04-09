import os
import time
import threading
import datetime
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from plotly.subplots import make_subplots
from gradio_client import Client
from supabase import create_client

app = FastAPI()

# --- CONFIG ---
URL_SB = os.getenv("URL_SB")
KEY_SB = os.getenv("KEY_SB")

try:
    supabase = create_client(URL_SB, KEY_SB)
    print("✅ Supabase conectado.", flush=True)
except Exception as e:
    supabase = None
    print(f"❌ Error Supabase: {e}", flush=True)

CATEGORIZED_TICKERS = {
    "Energy": {"Crude_Oil": "CL=F", "Brent_Oil": "BZ=F", "Natural_Gas": "NG=F", "Gasoline": "RB=F", "Heating_Oil": "HO=F", "Uranium": "URA"},
    "Metals": {"Copper": "HG=F", "Aluminum": "ALI=F", "Lithium": "LIT", "Steel": "SLX", "Gold": "GC=F", "Silver": "SI=F", "Platinum": "PL=F", "Palladium": "PA=F"},
    "Agriculture": {"Corn": "ZC=F", "Wheat": "ZW=F", "Soybeans": "ZS=F", "Coffee": "KC=F", "Sugar": "SB=F", "Cocoa": "CC=F", "Cotton": "CT=F"},
    "Macro_Crypto": {"SP500": "^GSPC", "VIX_Index": "^VIX", "Bitcoin": "BTC-USD", "Carbon_Credits": "KRBN"}
}
ALL_TICKERS = {k: v for cat in CATEGORIZED_TICKERS.values() for k, v in cat.items()}

def get_ai_prediction_v5(asset_name, prices_list):
    try:
        client = Client("marcosrgdata/trading-brain-v5")
        p_str = ",".join(map(str, prices_list[-48:]))
        return client.predict(asset_name=asset_name, prices_string=p_str, api_name="/predict_v5")
    except: return None

# --- WORKER BLINDADO (Arregla el error de 'Series') ---
def background_worker():
    while True:
        print(f"🤖 Ronda iniciada: {datetime.datetime.now()}", flush=True)
        for name, tid in ALL_TICKERS.items():
            try:
                # Bajamos 14d para que ETFs (Steel, Uranium) tengan >48 horas de datos
                df = yf.download(tid, period="14d", interval="1h", progress=False)
                
                if df.empty: continue

                # MÁGIA: Si yfinance devuelve MultiIndex (Series error), lo aplanamos
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(-1)
                
                df = df.dropna()
                
                # Extraemos valores como escalares puros para evitar el error 'Series'
                lp = float(df['Close'].iloc[-1])
                
                if supabase:
                    # 1. SIEMPRE GUARDAR HISTORIAL (Suben las 20.440 filas)
                    ma20 = df['Close'].tail(20).mean()
                    supabase.table("precios_historicos").insert({
                        "activo": name, "precio": round(lp, 2), 
                        "tendencia": "BULLISH" if lp > ma20 else "BEARISH"
                    }).execute()

                    # 2. IA Y RANGOS REALES
                    r_max = float(df['High'].tail(24).max())
                    r_min = float(df['Low'].tail(24).min())
                    
                    ai_trend, ai_conf, ai_max, ai_min = "N/A", "0%", 0, 0
                    
                    if len(df) >= 48:
                        ai = get_ai_prediction_v5(name, df['Close'].tolist())
                        if ai:
                            ai_trend, ai_conf = ai['prediction'], ai['confidence']
                            ai_max, ai_min = float(ai['expected_max']), float(ai['expected_min'])
                    
                    supabase.table("ai_predictions_v5").upsert({
                        "asset": name, "trend": ai_trend, "confidence": ai_conf,
                        "target_max": ai_max, "target_min": ai_min,
                        "real_max_24h": r_max, "real_min_24h": r_min
                    }).execute()
                
                print(f"✅ {name} ok.", flush=True)
                time.sleep(1.2)
            except Exception as e:
                print(f"❌ Error en {name}: {e}", flush=True)
        
        print("☕ Ronda terminada. Esperando 15 min.", flush=True)
        time.sleep(900)

threading.Thread(target=background_worker, daemon=True).start()

# --- DASHBOARD ---
@app.get("/visual-dashboard", response_class=HTMLResponse)
def get_dashboard():
    try:
        ticker_ids = list(ALL_TICKERS.values())
        raw_data = yf.download(ticker_ids, period="14d", interval="1h", group_by='ticker', threads=False)
        
        ai_data = {}
        if supabase:
            res = supabase.table("ai_predictions_v5").select("*").execute()
            ai_data = {item['asset']: item for item in res.data}

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07, row_heights=[0.7, 0.3],
                            subplot_titles=("V5.0 PRO QUANT TERMINAL", "MOMENTUM (RSI 14)"))
        
        market_data = []
        colors = {"Energy": "#ef4444", "Metals": "#f59e0b", "Agriculture": "#10b981", "Macro_Crypto": "#3b82f6"}
        trace_idx = 0

        for sector, assets in CATEGORIZED_TICKERS.items():
            for name, tid in assets.items():
                try:
                    # Arreglo para MultiIndex en el download del Dashboard
                    hist = raw_data[tid].dropna()
                    if isinstance(hist.columns, pd.MultiIndex):
                        hist.columns = hist.columns.get_level_values(-1)
                    
                    if len(hist) < 24: continue
                    
                    t7d = hist.index[-1] - pd.Timedelta(days=7)
                    idx7 = hist.index.get_indexer([t7d], method='nearest')[0]
                    perf = round(((hist['Close'].iloc[-1] / hist['Close'].iloc[idx7]) - 1) * 100, 2)
                    
                    fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'],
                                                 name=name, legendgroup=name, increasing_line_color=colors[sector], decreasing_line_color='#ffffff'), row=1, col=1)
                    
                    delta = hist['Close'].diff(); g = delta.where(delta > 0, 0).rolling(14).mean(); l = -delta.where(delta < 0, 0).rolling(14).mean()
                    rsi = 100 - (100 / (1 + (g / (l + 1e-9))))
                    fig.add_trace(go.Scatter(x=hist.index, y=rsi, showlegend=False, legendgroup=name, line=dict(color=colors[sector], width=1), opacity=0.3), row=2, col=1)

                    db = ai_data.get(name, {})
                    market_data.append({
                        "Asset": name, "Sector": sector, "Price": round(hist['Close'].iloc[-1], 2), "Perf": perf,
                        "R_Max": db.get('real_max_24h', 0), "R_Min": db.get('real_min_24h', 0),
                        "Trend": db.get('trend', 'N/A'), "Conf": db.get('confidence', '0%'),
                        "T_Max": db.get('target_max', 0), "T_Min": db.get('target_min', 0)
                    })
                    trace_idx += 2
                except: continue

        btns = [dict(method="restyle", label="GLOBAL VIEW", args=[{"visible": [True] * trace_idx}])]
        for s_name in CATEGORIZED_TICKERS.keys():
            vis = []
            for s, a in CATEGORIZED_TICKERS.items():
                for _ in a: vis.extend([(s == s_name)] * 2)
            btns.append(dict(method="restyle", label=s_name.upper(), args=[{"visible": vis}]))

        fig.update_layout(template="plotly_dark", height=850, margin=dict(t=150, b=5