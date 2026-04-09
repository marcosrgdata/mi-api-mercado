import os
import time
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
try:
    supabase = create_client(URL_SB, KEY_SB)
except:
    supabase = None

# --- ASSET LIST ---
CATEGORIZED_TICKERS = {
    "Energy": {"Crude_Oil": "CL=F", "Brent_Oil": "BZ=F", "Natural_Gas": "NG=F", "Gasoline": "RB=F", "Heating_Oil": "HO=F", "Uranium": "URA"},
    "Metals": {"Copper": "HG=F", "Aluminum": "ALI=F", "Lithium": "LIT", "Steel": "SLX", "Gold": "GC=F", "Silver": "SI=F", "Platinum": "PL=F", "Palladium": "PA=F"},
    "Agriculture": {"Corn": "ZC=F", "Wheat": "ZW=F", "Soybeans": "ZS=F", "Coffee": "KC=F", "Sugar": "SB=F", "Cocoa": "CC=F", "Cotton": "CT=F"},
    "Macro_Crypto": {"SP500": "^GSPC", "VIX_Index": "^VIX", "Bitcoin": "BTC-USD", "Carbon_Credits": "KRBN"}
}
ALL_TICKERS = {k: v for cat in CATEGORIZED_TICKERS.values() for k, v in cat.items()}

# --- AI ENGINES ---

def get_ai_prediction_v5(asset_name, prices_list):
    """LSTM Engine from Hugging Face."""
    try:
        if len(prices_list) < 48: return None
        client = Client("marcosrgdata/trading-brain-v5")
        prices_string = ",".join(map(str, prices_list[-48:]))
        return client.predict(asset_name=asset_name, prices_string=prices_string, api_name="/predict_v5")
    except: return None

def get_quant_prediction_v4(df, hours_ahead=24):
    """Random Forest Engine (for the dotted lines on the chart)."""
    data = df.copy()
    data['h'] = data.index.hour
    data['d'] = data.index.dayofweek
    data['ret'] = data['Close'].pct_change().fillna(0)
    X = data[['h', 'd', 'ret']].values
    y = data['Close'].values
    model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    model.fit(X, y)
    volat = data['Close'].diff().std()
    avg_p = data['Close'].tail(50).mean()
    preds = []
    curr = y[-1]
    for i in range(1, hours_ahead + 1):
        ai_dir = model.predict([[ (data.index[-1].hour + i)%24, data.index[-1].dayofweek, data['ret'].iloc[-1] ]])[0]
        noise = np.random.normal(0, volat)
        elasticity = (avg_p - curr) * 0.05
        curr = (ai_dir * 0.3) + (curr * 0.7) + noise + elasticity
        preds.append(y[-1] + (curr - y[-1]) * (0.96 ** i))
    return preds

# --- BACKGROUND WORKER ---
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
                time.sleep(1.2)
            except: continue
        time.sleep(900)

threading.Thread(target=background_worker, daemon=True).start()

# --- DASHBOARD ---
@app.get("/visual-dashboard", response_class=HTMLResponse)
def get_dashboard():
    try:
        ticker_ids = list(ALL_TICKERS.values())
        raw_data = yf.download(ticker_ids, period="14d", interval="1h", group_by='ticker', threads=True)
        
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
                    
                    # 1. Real Trace
                    fig.add_trace(go.Scatter(x=hist.index, y=perf, name=name, legendgroup=name, line=dict(color=colors[sector], width=2.5)), row=1, col=1)
                    
                    # 2. V4.6 Forecast Lines (Dotted)
                    proj_raw = get_quant_prediction_v4(hist)
                    proj_perf = [((v / base_p) - 1) * 100 for v in proj_raw]
                    f_idx = pd.date_range(start=hist.index[-1], periods=25, freq='h')[1:]
                    fig.add_trace(go.Scatter(x=[hist.index[-1]] + list(f_idx), y=[perf.iloc[-1]] + list(proj_perf),
                                             name=f"{name} Forecast", legendgroup=name, showlegend=False,
                                             line=dict(color=colors[sector], width=2, dash='dot'), opacity=0.4), row=1, col=1)
                    
                    # 3. RSI Trace
                    d = hist['Close'].diff(); g = d.where(d > 0, 0).rolling(14).mean(); l = -d.where(d < 0, 0).rolling(14).mean()
                    rsi = 100 - (100 / (1 + (g/(l + 1e-9))))
                    fig.add_trace(go.Scatter(x=hist.index, y=rsi, showlegend=False, legendgroup=name, line=dict(color=colors[sector], width=1, dash='dot'), opacity=0.2), row=2, col=1)

                    # 4. V5.0 LSTM Data for Table
                    ai_v5 = get_ai_prediction_v5(name, hist['Close'].tolist())
                    
                    market_data.append({
                        "Asset": name, "Sector": sector, "Price": round(hist['Close'].iloc[-1], 2), "Perf": round(perf.iloc[-1], 2),
                        "AI_Trend": ai_v5['prediction'] if ai_v5 else "N/A",
                        "AI_Conf": ai_v5['confidence'] if ai_v5 else "N/A",
                        "AI_Max": ai_v5['expected_max'] if ai_v5 else 0,
                        "AI_Min": ai_v5['expected_min'] if ai_v5 else 0
                    })
                    trace_idx += 3
                except: continue

        # --- RESTORE V4.6 UI BUTTONS ---
        btns = [dict(method="restyle", label="GLOBAL VIEW", args=[{"visible": [True] * trace_idx}])]
        for s_name in CATEGORIZED_TICKERS.keys():
            vis = []
            for s, a in CATEGORIZED_TICKERS.items():
                for _ in a:
                    v = (s == s_name); vis.extend([v, v, v])
            btns.append(dict(method="restyle", label=s_name.upper(), args=[{"visible": vis}]))

        fig.update_layout(template="plotly_dark", height=850, margin=dict(t=150, b=50), paper_bgcolor="#0a0a0a", plot_bgcolor="#0a0a0a",
                          legend=dict(itemclick="toggleothers", itemdoubleclick="toggle", font=dict(size=10), orientation="v", x=1.02, y=0.5),
                          updatemenus=[dict(type="buttons", direction="right", x=0.5, y=1.18, xanchor="center", buttons=btns, bgcolor="#1e293b", font=dict(color="white"), active=-1)])

        # --- CSS & TABLE ---
        css = """
        <style>
            rect.updatemenu-item-rect { fill: #1e293b !important; } 
            rect.updatemenu-item-rect:hover { fill: #334155 !important; } 
            rect.updatemenu-item-rect.active { fill: #2563eb !important; } 
            text.updatemenu-item-text { fill: #ffffff !important; font-weight: bold !important; } 
            table { width: 100%; border-collapse: collapse; color: white; table-layout: fixed; font-family: sans-serif; } 
            th { padding: 15px; text-align: left; background-color: #111827; color: #94a3b8; } 
            td { padding: 12px; border-bottom: 1px solid #1f2937; }
            .up { color: #10b981; font-weight: bold; }
            .down { color: #ef4444; font-weight: bold; }
            .conf { color: #3b82f6; font-size: 0.85em; }
        </style>
        """
        
        df_m = pd.DataFrame(market_data).sort_values(by="Perf", ascending=False)
        rows = ""
        for _, r in df_m.iterrows():
            t_class = "up" if r['AI_Trend'] == "UP" else "down"
            rows += f"""
            <tr>
                <td style='padding:12px; font-weight:bold;'>{r['Asset']}</td>
                <td style='color:#4b5563;'>{r['Sector']}</td>
                <td>${r['Price']}</td>
                <td class="{'up' if r['Perf']>0 else 'down'}">{r['Perf']}%</td>
                <td class="{t_class}">{r['AI_Trend']} <span class="conf">({r['AI_Conf']})</span></td>
                <td><span class="up">H: {r['AI_Max']}</span><br><span class="down">L: {r['AI_Min']}</span></td>
            </tr>
            """
        
        return HTMLResponse(content=f"<html><head>{css}</head><body style='margin:0; background:#0a0a0a;'>{fig.to_html(full_html=False, include_plotlyjs='cdn')}<div style='background:#0a0a0a; padding:40px;'><h2 style='text-align:center; color:#64748b; letter-spacing: 2px; font-family:sans-serif;'>V5.0 INSTITUTIONAL SUMMARY</h2><table><thead><tr><th>ASSET</th><th>SECTOR</th><th>PRICE</th><th>7D PERF</th><th>AI TREND</th><th>AI TARGET RANGE (24H)</th></tr></thead><tbody>{rows}</tbody></table></div></body></html>")
    except Exception as e:
        return HTMLResponse(content=f"<h1>Error</h1><code>{str(e)}</code>", status_code=500)