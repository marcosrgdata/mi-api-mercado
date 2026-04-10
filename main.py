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
except:
    supabase = None

CATEGORIZED_TICKERS = {
    "Energy": {"Crude_Oil": "CL=F", "Brent_Oil": "BZ=F", "Natural_Gas": "NG=F", "Gasoline": "RB=F", "Heating_Oil": "HO=F", "Uranium": "URA"},
    "Metals": {"Copper": "HG=F", "Aluminum": "ALI=F", "Lithium": "LIT", "Steel": "SLX", "Gold": "GC=F", "Silver": "SI=F", "Platinum": "PL=F", "Palladium": "PA=F"},
    "Agriculture": {"Corn": "ZC=F", "Wheat": "ZW=F", "Soybeans": "ZS=F", "Coffee": "KC=F", "Sugar": "SB=F", "Cocoa": "CC=F", "Cotton": "CT=F"},
    "Macro_Crypto": {"SP500": "^GSPC", "VIX_Index": "^VIX", "Bitcoin": "BTC-USD", "Carbon_Credits": "KRBN"}
}
ALL_TICKERS = {k: v for cat in CATEGORIZED_TICKERS.values() for k, v in cat.items()}

# --- UTILS ---
def clean_df(df):
    if df.empty: return df
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    if isinstance(df['Close'], pd.DataFrame): df = df.loc[:, ~df.columns.duplicated()]
    return df

def get_ai_prediction_v5(asset_name, prices_list):
    try:
        client = Client("marcosrgdata/trading-brain-v5")
        p_str = ",".join(map(str, prices_list[-48:]))
        result = client.predict(asset_name=asset_name, prices_string=p_str, api_name="/predict_v5")
        # Validamos que la respuesta tenga las llaves necesarias
        if result and isinstance(result, dict) and 'expected_max' in result:
            return result
        return None
    except:
        return None

# --- WORKER V5.4 (FIX KEYERROR COTTON) ---
def background_worker():
    while True:
        print(f"🤖 Ciclo iniciado: {datetime.datetime.now()}", flush=True)
        for name, tid in ALL_TICKERS.items():
            try:
                # Cotton necesita más histórico para rellenar huecos
                p = "20d" if name == "Cotton" else "14d"
                df = yf.download(tid, period=p, interval="1h", progress=False)
                df = clean_df(df).dropna()
                if df.empty: continue
                lp = float(df['Close'].iloc[-1])
                
                if supabase:
                    # 1. Histórico (Filas 20.440+)
                    ma20 = df['Close'].tail(20).mean()
                    supabase.table("precios_historicos").insert({"activo": name, "precio": round(lp, 2), "tendencia": "BULLISH" if lp > ma20 else "BEARISH"}).execute()

                    # 2. IA y Logs con SEGURIDAD EXTRA
                    ai = get_ai_prediction_v5(name, df['Close'].tolist())
                    
                    # RANGOS REALES SIEMPRE SE GUARDAN
                    r_max = round(float(df['High'].tail(24).max()), 2)
                    r_min = round(float(df['Low'].tail(24).min()), 2)
                    
                    # Si la IA responde correctamente, usamos sus datos. Si no, N/A.
                    ai_trend = ai.get('prediction', 'N/A') if ai else "N/A"
                    ai_conf = ai.get('confidence', '0%') if ai else "0%"
                    ai_max = round(float(ai.get('expected_max', 0)), 2) if ai else 0
                    ai_min = round(float(ai.get('expected_min', 0)), 2) if ai else 0
                    
                    supabase.table("ai_predictions_v5").upsert({
                        "asset": name, "trend": ai_trend, "confidence": ai_conf,
                        "target_max": ai_max, "target_min": ai_min, "real_max_24h": r_max, "real_min_24h": r_min
                    }).execute()

                    # Solo logueamos si hay predicción real
                    if ai:
                        supabase.table("ai_prediction_logs").insert({
                            "asset": name, "trend": ai_trend, "target_max": ai_max, "target_min": ai_min
                        }).execute()
                
                print(f"✅ {name}: Procesado.", flush=True)
                time.sleep(1.5)
            except Exception as e:
                print(f"❌ Error {name}: {str(e)}", flush=True)
        time.sleep(900)

threading.Thread(target=background_worker, daemon=True).start()

# --- DASHBOARD V5.4 (UI V4.6 STYLE) ---
@app.get("/visual-dashboard", response_class=HTMLResponse)
def get_dashboard():
    try:
        ticker_ids = list(ALL_TICKERS.values())
        raw_data = yf.download(ticker_ids, period="14d", interval="1h", group_by='ticker', threads=False)
        
        ai_current = {item['asset']: item for item in supabase.table("ai_predictions_v5").select("*").execute().data}
        
        # Validación: 18-30 horas atrás
        t_limit = (datetime.datetime.now() - datetime.timedelta(hours=18)).isoformat()
        t_start = (datetime.datetime.now() - datetime.timedelta(hours=30)).isoformat()
        logs_res = supabase.table("ai_prediction_logs").select("*").lt("created_at", t_limit).gt("created_at", t_start).order("created_at").execute()
        
        ai_past = {}
        for log in logs_res.data:
            if log['asset'] not in ai_past: ai_past[log['asset']] = log

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07, row_heights=[0.7, 0.3])
        market_data = []
        colors = {"Energy": "#ef4444", "Metals": "#f59e0b", "Agriculture": "#10b981", "Macro_Crypto": "#3b82f6"}
        trace_idx = 0

        for sector, assets in CATEGORIZED_TICKERS.items():
            for name, tid in assets.items():
                try:
                    hist = clean_df(raw_data[tid]).dropna()
                    if len(hist) < 24: continue
                    t7d = hist.index[-1] - pd.Timedelta(days=7); idx7 = hist.index.get_indexer([t7d], method='nearest')[0]
                    perf = round(((hist['Close'].iloc[-1] / hist['Close'].iloc[idx7]) - 1) * 100, 2)
                    
                    fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'],
                                                 name=name, legendgroup=name, increasing_line_color=colors[sector], decreasing_line_color='#ffffff'), row=1, col=1)
                    
                    curr, past = ai_current.get(name, {}), ai_past.get(name, {})
                    
                    # Validación
                    val_text = "Wait 24h..."
                    if past:
                        hit = (curr.get('real_max_24h', 0) <= past['target_max'] * 1.005) and (curr.get('real_min_24h', 0) >= past['target_min'] * 0.995)
                        val_text = "✅ SUCCESS" if hit else "❌ MISSED"

                    market_data.append({
                        "Asset": name, "Sector": sector, "Price": round(float(hist['Close'].iloc[-1]), 2), "Perf": perf,
                        "R_H": curr.get('real_max_24h', 0), "R_L": curr.get('real_min_24h', 0),
                        "Trend": curr.get('trend', 'N/A'), "Conf": curr.get('confidence', '0%'),
                        "T_Max": curr.get('target_max', 0), "T_Min": curr.get('target_min', 0),
                        "Past_Call": f"{past.get('trend', '-')}: [{past.get('target_min', 0)}-{past.get('target_max', 0)}]" if past else "Collecting...",
                        "Val": val_text
                    })
                    trace_idx += 2
                except: continue

        btns = [dict(method="restyle", label="GLOBAL VIEW", args=[{"visible": [True] * trace_idx}])]
        for s_name in CATEGORIZED_TICKERS.keys():
            vis = []
            for s, a in CATEGORIZED_TICKERS.items():
                for _ in a: vis.extend([(s == s_name)] * 2)
            btns.append(dict(method="restyle", label=s_name.upper(), args=[{"visible": vis}]))

        fig.update_layout(template="plotly_dark", height=800, margin=dict(t=150, b=50), paper_bgcolor="#0a0a0a", plot_bgcolor="#0a0a0a",
                          xaxis_rangeslider_visible=False, updatemenus=[dict(type="buttons", direction="right", x=0.5, y=1.18, xanchor="center", buttons=btns, bgcolor="#1e293b")])

        def render_bar(curr, t_min, t_max):
            if not t_max or t_max == t_min: return ""
            pos = max(0, min(100, ((curr - t_min) / (t_max - t_min)) * 100))
            color = "#ef4444" if pos > 80 else "#10b981" if pos < 20 else "#3b82f6"
            return f'<div style="width:100%; background:#1e293b; height:6px; border-radius:3px; margin-top:8px; position:relative;"><div style="position:absolute; left:{pos}%; width:10px; height:10px; background:{color}; border-radius:50%; top:-2px; box-shadow:0 0 5px {color};"></div></div>'

        css = "<style>body{background:#0a0a0a;color:white;font-family:sans-serif;}table{width:100%;border-collapse:collapse;table-layout:fixed;}th{padding:15px;text-align:left;background:#111827;color:#94a3b8;font-size:0.75em;}tr{border-bottom:1px solid #1f2937;}td{padding:12px;}.up{color:#10b981;font-weight:bold;}.down{color:#ef4444;font-weight:bold;}.val-box{border-left:1px solid #334155; padding-left:10px;}</style>"
        
        rows = ""
        for r in sorted(market_data, key=lambda x: x['Perf'], reverse=True):
            tc, pc = ("up" if r['Trend'] == "UP" else "down"), ("up" if r['Perf'] > 0 else "down")
            vc = "up" if "✅" in r['Val'] else "down" if "❌" in r['Val'] else ""
            rows += f"""
            <tr>
                <td><b>{r['Asset']}</b><br><small style='color:#4b5563'>{r['Sector']}</small></td>
                <td>${r['Price']}</td>
                <td class='{pc}'>{r['Perf']}%</td>
                <td style='color:#64748b'><small>Real High/Low:</small><br>{r['R_H']} / {r['R_L']}</td>
                <td><span class='{tc}'>{r['Trend']} ({r['Conf']})</span><br><small>Target: {r['T_Min']} - {r['T_Max']}</small>{render_bar(r['Price'], r['T_Min'], r['T_Max'])}</td>
                <td class="val-box"><small style='color:#94a3b8'>Yesterday's Call:</small><br><small>{r['Past_Call']}</small><br><b class='{vc}'>{r['Val']}</b></td>
            </tr>
            """
        return HTMLResponse(content=f"<html><head>{css}</head><body>{fig.to_html(full_html=False, include_plotlyjs='cdn')}<div style='padding:40px;'><h2 style='text-align:center;color:#64748b;letter-spacing:2px;'>V5.4 QUANT TERMINAL</h2><table><thead><tr><th>ASSET</th><th>PRICE</th><th>7D ROLL</th><th>REAL 24H (TODAY)</th><th>AI TARGET (TOMORROW)</th><th>AI VALIDATION (YESTERDAY)</th></tr></thead><tbody>{rows}</tbody></table></div></body></html>")
    except Exception as e: return HTMLResponse(content=f"Error: {e}")