import os
import time
import datetime
import threading
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from gradio_client import Client
from supabase import create_client

app = FastAPI()

# --- CONFIGURATION ---
URL_SB = os.getenv("URL_SB")
KEY_SB = os.getenv("KEY_SB")
supabase = create_client(URL_SB, KEY_SB) if URL_SB and KEY_SB else None

CATEGORIZED_TICKERS = {
    "Energy": {"Crude_Oil": "CL=F", "Brent_Oil": "BZ=F", "Natural_Gas": "NG=F", "Gasoline": "RB=F", "Heating_Oil": "HO=F", "Uranium": "URA"},
    "Metals": {"Copper": "HG=F", "Aluminum": "ALI=F", "Lithium": "LIT", "Steel": "SLX", "Gold": "GC=F", "Silver": "SI=F", "Platinum": "PL=F", "Palladium": "PA=F"},
    "Agriculture": {"Corn": "ZC=F", "Wheat": "ZW=F", "Soybeans": "ZS=F", "Coffee": "KC=F", "Sugar": "SB=F", "Cocoa": "CC=F", "Cotton": "CT=F"},
    "Macro_Crypto": {"SP500": "^GSPC", "VIX_Index": "^VIX", "Bitcoin": "BTC-USD", "Carbon_Credits": "KRBN"}
}
ALL_TICKERS = {k: v for cat in CATEGORIZED_TICKERS.values() for k, v in cat.items()}

# --- AI ENGINES ---

def get_ai_prediction_v5(asset_name, prices_list):
    """Calls the LSTM V5.0 model on Hugging Face."""
    try:
        if len(prices_list) < 48:
            return None
        # Taking only the last 48 hours
        input_prices = prices_list[-48:]
        client = Client("marcosrgdata/trading-brain-v5")
        prices_string = ",".join(map(str, input_prices))
        
        result = client.predict(
            asset_name=asset_name,
            prices_string=prices_string,
            api_name="/predict_v5"
        )
        return result
    except Exception as e:
        print(f"HF Connection Error: {e}")
        return None

def get_quant_prediction_v4(df, hours_ahead=24):
    """Hybrid AI: Random Forest + Stochastic Noise."""
    data = df.copy()
    data['h'] = data.index.hour
    data['d'] = data.index.dayofweek
    data['ret'] = data['Close'].pct_change().fillna(0)
    X = data[['h', 'd', 'ret']].values
    y = data['Close'].values
    model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
    model.fit(X, y)
    
    volat = data['Close'].diff().std()
    avg_p = data['Close'].tail(50).mean()
    preds = []
    curr = y[-1]
    for i in range(1, hours_ahead + 1):
        ai_dir = model.predict([[ (data.index[-1].hour + i)%24, data.index[-1].dayofweek, data['ret'].iloc[-1] ]])[0]
        noise = np.random.normal(0, volat * 0.5)
        elasticity = (avg_p - curr) * 0.05
        curr = (ai_dir * 0.3) + (curr * 0.7) + noise + elasticity
        preds.append(y[-1] + (curr - y[-1]) * (0.96 ** i))
    return preds

# --- WORKER ---
def background_worker():
    while True:
        for name, tid in ALL_TICKERS.items():
            try:
                t = yf.Ticker(tid); h = t.history(period="5d")
                if not h.empty and supabase:
                    lp = round(h['Close'].iloc[-1], 2)
                    ma = h['Close'].tail(20).mean()
                    tr = "BULLISH" if lp > ma else "BEARISH"
                    supabase.table("precios_historicos").insert({"activo": name, "precio": lp, "tendencia": tr}).execute()
                time.sleep(1)
            except: continue
        time.sleep(900)

threading.Thread(target=background_worker, daemon=True).start()

# --- DASHBOARD ---
@app.get("/", response_class=HTMLResponse)
def get_dashboard():
    try:
        ticker_ids = list(ALL_TICKERS.values())
        raw_data = yf.download(ticker_ids, period="14d", interval="1h", group_by='ticker')
        
        market_data = []
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3],
                            subplot_titles=("V5.0 PRO QUANT TERMINAL", "MOMENTUM (RSI 14)"))
        
        colors = {"Energy": "#ef4444", "Metals": "#f59e0b", "Agriculture": "#10b981", "Macro_Crypto": "#3b82f6"}
        trace_idx = 0

        for sector, assets in CATEGORIZED_TICKERS.items():
            for name, tid in assets.items():
                try:
                    hist = raw_data[tid].dropna() if len(ticker_ids) > 1 else raw_data.dropna()
                    if len(hist) < 48: continue 
                    
                    base_p = hist['Close'].iloc[0]
                    perf = ((hist['Close'] / base_p) - 1) * 100
                    
                    # 1. Real Price Trace
                    fig.add_trace(go.Scatter(x=hist.index, y=perf, name=name, legendgroup=name, line=dict(color=colors[sector], width=2)), row=1, col=1)
                    
                    # 2. V4.6 Projection (Visual dots)
                    proj_raw = get_quant_prediction_v4(hist)
                    proj_perf = [((v / base_p) - 1) * 100 for v in proj_raw]
                    f_idx = pd.date_range(start=hist.index[-1], periods=25, freq='h')[1:]
                    fig.add_trace(go.Scatter(x=[hist.index[-1]] + list(f_idx), y=[perf.iloc[-1]] + list(proj_perf),
                                             name=f"{name} Forecast", legendgroup=name, showlegend=False,
                                             line=dict(color=colors[sector], width=1, dash='dot'), opacity=0.4), row=1, col=1)
                    
                    # 3. RSI
                    d = hist['Close'].diff(); g = d.where(d > 0, 0).rolling(14).mean(); l = -d.where(d < 0, 0).rolling(14).mean()
                    rsi = 100 - (100 / (1 + (g/(l + 1e-9))))
                    fig.add_trace(go.Scatter(x=hist.index, y=rsi, showlegend=False, legendgroup=name, line=dict(color=colors[sector], width=1), opacity=0.3), row=2, col=1)

                    # 4. V5.0 HF Inference for Table
                    ai_v5 = get_ai_prediction_v5(name, hist['Close'].tolist())
                    
                    market_data.append({
                        "Asset": name, "Sector": sector, "Price": round(hist['Close'].iloc[-1], 2),
                        "Perf": round(perf.iloc[-1], 2), 
                        "AI_Trend": ai_v5['prediction'] if ai_v5 else "N/A",
                        "AI_Conf": ai_v5['confidence'] if ai_v5 else "N/A",
                        "AI_Max": ai_v5['expected_max'] if ai_v5 else 0,
                        "AI_Min": ai_v5['expected_min'] if ai_v5 else 0
                    })
                    trace_idx += 3
                except: continue

        # UI & TABLE
        css = "<style>body{background:#0a0a0a; color:white; font-family:sans-serif;} table{width:100%; border-collapse:collapse;} th{background:#111827; padding:15px; text-align:left; color:#94a3b8;} td{padding:12px; border-bottom:1px solid #1f2937;} .up{color:#10b981;} .down{color:#ef4444;} .conf{color:#3b82f6; font-size:0.8em;}</style>"
        
        rows = ""
        for r in market_data:
            trend_class = "up" if r['AI_Trend'] == "UP" else "down"
            rows += f"<tr><td><b>{r['Asset']}</b></td><td>{r['Sector']}</td><td>${r['Price']}</td>"
            rows += f"<td>{r['Perf']}%</td><td class='{trend_class}'>{r['AI_Trend']} <span class='conf'>({r['AI_Conf']})</span></td>"
            rows += f"<td><span class='up'>H: {r['AI_Max']}</span><br><span class='down'>L: {r['AI_Min']}</span></td></tr>"
        
        return HTMLResponse(content=f"<html><head>{css}</head><body>{fig.to_html(full_html=False, include_plotlyjs='cdn')}<div style='padding:40px;'><h2>V5.0 DEEP LEARNING SUMMARY</h2><table><thead><tr><th>ASSET</th><th>SECTOR</th><th>LIVE PRICE</th><th>7D PERF</th><th>AI TREND</th><th>AI TARGET RANGE (24H)</th></tr></thead><tbody>{rows}</tbody></table></div></body></html>")
    except Exception as e:
        return HTMLResponse(content=f"<h1>Error</h1><code>{str(e)}</code>", status_code=500)