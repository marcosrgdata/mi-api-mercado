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
    print(f"❌ Error conexión Supabase: {e}", flush=True)

CATEGORIZED_TICKERS = {
    "Energy": {"Crude_Oil": "CL=F", "Brent_Oil": "BZ=F", "Natural_Gas": "NG=F", "Gasoline": "RB=F", "Heating_Oil": "HO=F", "Uranium": "URA"},
    "Metals": {"Copper": "HG=F", "Aluminum": "ALI=F", "Lithium": "LIT", "Steel": "SLX", "Gold": "GC=F", "Silver": "SI=F", "Platinum": "PL=F", "Palladium": "PA=F"},
    "Agriculture": {"Corn": "ZC=F", "Wheat": "ZW=F", "Soybeans": "ZS=F", "Coffee": "KC=F", "Sugar": "SB=F", "Cocoa": "CC=F", "Cotton": "CT=F"},
    "Macro_Crypto": {"SP500": "^GSPC", "VIX_Index": "^VIX", "Bitcoin": "BTC-USD", "Carbon_Credits": "KRBN"}
}
ALL_TICKERS = {k: v for cat in CATEGORIZED_TICKERS.values() for k, v in cat.items()}

def get_ai_prediction_v5(asset_name, prices_list):
    try:
        # Timeout corto para que no bloquee el hilo si falla
        client = Client("marcosrgdata/trading-brain-v5")
        p_str = ",".join(map(str, prices_list[-48:]))
        return client.predict(asset_name=asset_name, prices_string=p_str, api_name="/predict_v5")
    except: return None

# --- WORKER DE SEGURIDAD (Historial primero, IA después) ---
def background_worker():
    while True:
        print(f"🔄 Iniciando ronda de datos: {datetime.datetime.now()}", flush=True)
        for name, tid in ALL_TICKERS.items():
            try:
                # 1. Descarga rápida
                df = yf.download(tid, period="5d", interval="1h", progress=False).dropna()
                if df.empty: continue
                
                lp = round(float(df['Close'].iloc[-1]), 2)
                
                if supabase:
                    # --- PASO A: INSERTAR HISTÓRICO (PRIORIDAD) ---
                    # Esto es lo que hace subir el contador de filas (20440+)
                    ma20 = df['Close'].tail(20).mean()
                    trend_label = "BULLISH" if lp > ma20 else "BEARISH"
                    
                    supabase.table("precios_historicos").insert({
                        "activo": name, 
                        "precio": lp, 
                        "tendencia": trend_label
                    }).execute()
                    print(f"📈 {name}: Precio guardado en historial.", flush=True)

                    # --- PASO B: ACTUALIZAR IA (EXTRAS) ---
                    # Lo intentamos, pero si falla no rompe el bucle
                    try:
                        ai = get_ai_prediction_v5(name, df['Close'].tolist())
                        if ai:
                            supabase.table("ai_predictions_v5").upsert({
                                "asset": name,
                                "trend": ai['prediction'],
                                "confidence": ai['confidence'],
                                "target_max": float(ai['expected_max']),
                                "target_min": float(ai['expected_min']),
                                "real_max_24h": float(df['High'].tail(24).max()),
                                "real_min_24h": float(df['Low'].tail(24).min())
                            }).execute()
                    except:
                        print(f"⚠️ {name}: IA no disponible en este momento.", flush=True)

                time.sleep(1.2) # Evitar bloqueos de IP
            except Exception as e:
                print(f"⚠️ Error en {name}: {e}", flush=True)
        
        print("☕ Ronda terminada. Esperando 15 minutos.", flush=True)
        time.sleep(900)

threading.Thread(target=background_worker, daemon=True).start()

# --- DASHBOARD UI (V4.6 Style con Mejoras) ---
@app.get("/visual-dashboard", response_class=HTMLResponse)
def get_dashboard():
    try:
        ticker_ids = list(ALL_TICKERS.values())
        raw_data = yf.download(ticker_ids, period="14d", interval="1h", group_by='ticker', threads=True)
        
        # Leemos las predicciones para la tabla
        ai_data = {}
        if supabase:
            res = supabase.table("ai_predictions_v5").select("*").execute()
            ai_data = {item['asset']: item for item in res.data}

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3],
                            subplot_titles=("V5.0 TERMINAL (V4.6 UI)", "MOMENTUM"))
        
        market_data = []
        colors = {"Energy": "#ef4444", "Metals": "#f59e0b", "Agriculture": "#10b981", "Macro_Crypto": "#3b82f6"}
        trace_idx = 0

        for sector, assets in CATEGORIZED_TICKERS.items():
            for name, tid in assets.items():
                try:
                    hist = raw_data[tid].dropna() if len(ticker_ids) > 1 else raw_data.dropna()
                    if len(hist) < 24: continue
                    
                    # 7D Rolling Perf
                    t7d = hist.index[-1] - pd.Timedelta(days=7)
                    idx7 = hist.index.get_indexer([t7d], method='nearest')[0]
                    p7 = round(((hist['Close'].iloc[-1] / hist['Close'].iloc[idx7]) - 1) * 100, 2)
                    
                    # Candlesticks
                    fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'],
                                                 name=name, legendgroup=name, increasing_line_color=colors[sector], decreasing_line_color='#ffffff'), row=1, col=1)
                    
                    # RSI
                    d = hist['Close'].diff(); g = d.where(d > 0, 0).rolling(14).mean(); l = -d.where(d < 0, 0).rolling(14).mean()
                    rsi = 100 - (100 / (1 + (g/(l + 1e-9))))
                    fig.add_trace(go.Scatter(x=hist.index, y=rsi, showlegend=False, legendgroup=name, line=dict(color=colors[sector], width=1), opacity=0.3), row=2, col=1)

                    db = ai_data.get(name, {})
                    market_data.append({
                        "Asset": name, "Sector": sector, "Price": round(hist['Close'].iloc[-1], 2), "Perf": p7,
                        "T_Trend": db.get('trend', 'N/A'), "T_Conf": db.get('confidence', '0%'),
                        "T_Max": db.get('target_max', 0), "T_Min": db.get('target_min', 0)
                    })
                    trace_idx += 2
                except: continue

        # --- BOTONES Y LEYENDA (ESTILO V4.6) ---
        btns = [dict(method="restyle", label="GLOBAL VIEW", args=[{"visible": [True] * trace_idx}])]
        for s_name in CATEGORIZED_TICKERS.keys():
            vis = []
            for s, a in CATEGORIZED_TICKERS.items():
                for _ in a: vis.extend([(s == s_name)] * 2)
            btns.append(dict(method="restyle", label=s_name.upper(), args=[{"visible": vis}]))

        fig.update_layout(template="plotly_dark", height=850, margin=dict(t=150, b=50), paper_bgcolor="#0a0a0a", plot_bgcolor="#0a0a0a",
                          xaxis_rangeslider_visible=False,
                          legend=dict(itemclick="toggleothers", itemdoubleclick="toggle", font=dict(size=10), orientation="v", x=1.02, y=0.5),
                          updatemenus=[dict(type="buttons", direction="right", x=0.5, y=1.18, xanchor="center", buttons=btns, bgcolor="#1e293b", font=dict(color="white"), active=-1)])

        css = "<style>rect.updatemenu-item-rect { fill: #1e293b !important; } rect.updatemenu-item-rect.active { fill: #2563eb !important; } table { width: 100%; border-collapse: collapse; color: white; font-family: sans-serif; } th { text-align: left; padding: 15px; background: #111827; color: #94a3b8; } td { padding: 12px; border-bottom: 1px solid #1f2937; } .up { color: #10b981; font-weight: bold; } .down { color: #ef4444; font-weight: bold; }</style>"
        
        rows = ""
        for r in sorted(market_data, key=lambda x: x['Perf'], reverse=True):
            tc = "up" if r['T_Trend'] == "UP" else "down"
            pc = "up" if r['Perf'] > 0 else "down"
            rows += f"<tr><td><b>{r['Asset']}</b></td><td>${r['Price']}</td><td class='{pc}'>{r['Perf']}%</td><td class='{tc}'>{r['T_Trend']} ({r['T_Conf']})</td><td>L: {r['T_Min']}<br>H: {r['T_Max']}</td></tr>"
        
        return HTMLResponse(content=f"<html><head>{css}</head><body style='margin:0; background:#0a0a0a;'>{fig.to_html(full_html=False, include_plotlyjs='cdn')}<div style='padding:40px;'><table><thead><tr><th>ASSET</th><th>PRICE</th><th>7D ROLL</th><th>AI TREND</th><th>AI TARGET</th></tr></thead><tbody>{rows}</tbody></table></div></body></html>")
    except Exception as e:
        return HTMLResponse(content=f"<h1>Error</h1><code>{str(e)}</code>", status_code=500)