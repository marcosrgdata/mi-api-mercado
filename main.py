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
        client = Client("marcosrgdata/trading-brain-v5")
        p_str = ",".join(map(str, prices_list[-48:]))
        return client.predict(asset_name=asset_name, prices_string=p_str, api_name="/predict_v5")
    except: return None

# --- WORKER TOTAL (Históricos + IA) ---
def background_worker():
    while True:
        print(f"🤖 Ciclo iniciado: {datetime.datetime.now()}", flush=True)
        for name, tid in ALL_TICKERS.items():
            try:
                # 1. Descarga datos
                df = yf.download(tid, period="7d", interval="1h", progress=False).dropna()
                if len(df) < 48: continue
                
                lp = round(float(df['Close'].iloc[-1]), 2)
                
                if supabase:
                    # A. INSERTAR EN HISTÓRICO (Para que suban las 20.440 filas)
                    ma20 = df['Close'].tail(20).mean()
                    trend_label = "BULLISH" if lp > ma20 else "BEARISH"
                    supabase.table("precios_historicos").insert({
                        "activo": name, "precio": lp, "tendencia": trend_label
                    }).execute()

                    # B. ACTUALIZAR PREDICCIÓN IA (Para el Dashboard rápido)
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
                
                print(f"✅ {name} procesado (Precio e IA)", flush=True)
                time.sleep(1.5)
            except Exception as e:
                print(f"⚠️ Error en {name}: {e}", flush=True)
        
        print("☕ Fin del ciclo. Esperando 15 min.", flush=True)
        time.sleep(900)

threading.Thread(target=background_worker, daemon=True).start()

# --- UI RENDERER ---
def render_thermo(curr, t_min, t_max):
    if not t_max or t_max == t_min: return ""
    pos = max(0, min(100, ((curr - t_min) / (t_max - t_min)) * 100))
    color = "#ef4444" if pos > 80 else "#10b981" if pos < 20 else "#3b82f6"
    return f'<div style="width:100%; background:#1e293b; height:6px; border-radius:3px; margin-top:8px; position:relative;"><div style="position:absolute; left:{pos}%; width:10px; height:10px; background:{color}; border-radius:50%; top:-2px; box-shadow:0 0 5px {color};"></div></div>'

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
                    hist = raw_data[tid].dropna() if len(ticker_ids) > 1 else raw_data.dropna()
                    if len(hist) < 48: continue
                    
                    target_7d = hist.index[-1] - pd.Timedelta(days=7)
                    idx_7d = hist.index.get_indexer([target_7d], method='nearest')[0]
                    perf = round(((hist['Close'].iloc[-1] / hist['Close'].iloc[idx_7d]) - 1) * 100, 2)
                    
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

        fig.update_layout(template="plotly_dark", height=850, margin=dict(t=150, b=50), paper_bgcolor="#0a0a0a", plot_bgcolor="#0a0a0a",
                          xaxis_rangeslider_visible=False,
                          legend=dict(itemclick="toggleothers", itemdoubleclick="toggle", font=dict(size=10), orientation="v", x=1.02, y=0.5),
                          updatemenus=[dict(type="buttons", direction="right", x=0.5, y=1.18, xanchor="center", buttons=btns, bgcolor="#1e293b", font=dict(color="white"), active=-1)])

        css = "<style>rect.updatemenu-item-rect { fill: #1e293b !important; } rect.updatemenu-item-rect:hover { fill: #334155 !important; } rect.updatemenu-item-rect.active { fill: #2563eb !important; } text.updatemenu-item-text { fill: #ffffff !important; font-weight: bold !important; } table { width: 100%; border-collapse: collapse; color: white; table-layout: fixed; font-family: sans-serif; } th { padding: 15px; text-align: left; background-color: #111827; color: #94a3b8; font-size: 0.8em; } tr { border-bottom: 1px solid #1f2937; } .up { color: #10b981; font-weight: bold; } .down { color: #ef4444; font-weight: bold; }</style>"
        
        rows = ""
        for r in sorted(market_data, key=lambda x: x['Perf'], reverse=True):
            t_col = "up" if r['Trend'] == "UP" else "down"
            p_col = "up" if r['Perf'] > 0 else "down"
            rows += f"""
            <tr>
                <td style='padding:12px;'><b>{r['Asset']}</b><br><small style='color:#4b5563'>{r['Sector']}</small></td>
                <td>${r['Price']}</td>
                <td class='{p_col}'>{r['Perf']}%</td>
                <td style='color:#64748b'>MAX: ${r['R_Max']}<br>MIN: ${r['R_Min']}</td>
                <td class='{t_col}'>{r['Trend']} <small style='color:#3b82f6;'>({r['Conf']})</small></td>
                <td style='width:220px; padding:10px;'>
                    <div style='display:flex; justify-content:space-between; font-size:0.7em;'><span class='down'>L: {r['T_Min']}</span><span class='up'>H: {r['T_Max']}</span></div>
                    {render_thermo(r['Price'], r['T_Min'], r['T_Max'])}
                </td>
            </tr>
            """
        return HTMLResponse(content=f"<html><head>{css}</head><body style='margin:0; background:#0a0a0a;'>{fig.to_html(full_html=False, include_plotlyjs='cdn')}<div style='background:#0a0a0a; padding:40px;'><h2 style='text-align:center; color:#64748b; letter-spacing: 2px; font-family:sans-serif;'>V5.0 INSTITUTIONAL MONITOR</h2><table><thead><tr><th>ASSET</th><th>PRICE</th><th>7D ROLL %</th><th>REAL 24H RANGE</th><th>AI SIGNAL</th><th>AI POSITIONING</th></tr></thead><tbody>{rows}</tbody></table></div></body></html>")
    except Exception as e:
        return HTMLResponse(content=f"<h1>Error</h1><code>{str(e)}</code>", status_code=500)