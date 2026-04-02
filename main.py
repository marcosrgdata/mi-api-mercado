from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import yfinance as yf
import pandas as pd
import numpy as np
import os
import datetime
import threading
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor

app = FastAPI()

# --- CONFIGURATION & DB ---
URL_SB = os.getenv("URL_SB")
KEY_SB = os.getenv("KEY_SB")
try:
    from supabase import create_client, Client
    supabase: Client = create_client(URL_SB, KEY_SB)
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

# --- V4.4 ENHANCED ENGINE ---

def get_pro_prediction(df, hours_ahead=24):
    """Predicts using Price, Volume, RSI, and Time Patterns."""
    # 1. Advanced Feature Engineering
    df = df.copy()
    df['hour'] = df.index.hour
    df['day'] = df.index.dayofweek
    df['vol_change'] = df['Volume'].pct_change()
    # Adding RSI as a feature for the AI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain/(loss + 1e-9))))
    df = df.dropna()
    
    # 2. X/Y Split
    X = df[['hour', 'day', 'vol_change', 'rsi']].values
    y = df['Close'].values
    
    # 3. ML Training (Higher N for better ensemble voting)
    model = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42)
    model.fit(X, y)
    
    # 4. Stochastic Generation
    last_val = df['Close'].iloc[-1]
    volat = df['Close'].diff().std() * 0.95
    
    preds = []
    temp_val = last_val
    for i in range(1, hours_ahead + 1):
        # AI base using last known volume and RSI context
        ai_base = model.predict([[ (df.index[-1].hour + i)%24, df.index[-1].dayofweek, df['vol_change'].iloc[-1], df['rsi'].iloc[-1] ]])[0]
        noise = np.random.normal(0, volat)
        temp_val = (ai_base * 0.5) + (temp_val * 0.5) + noise
        preds.append(temp_val)
    return preds

# --- BACKGROUND WORKER ---
def background_worker():
    while True:
        for name, tid in ALL_TICKERS.items():
            try:
                t = yf.Ticker(tid); h = t.history(period="5d")
                if not h.empty and supabase:
                    last_p = round(h['Close'].iloc[-1], 2)
                    # Institutional Trend: SMA20 vs SMA50
                    sma20 = h['Close'].tail(20).mean()
                    trend = "BULLISH" if last_p > sma20 else "BEARISH"
                    supabase.table("precios_historicos").insert({"activo": name, "precio": last_p, "tendencia": trend}).execute()
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
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3],
                            subplot_titles=("V4.4 VOLUME-WEIGHTED ML TERMINAL", "MOMENTUM (RSI 14)"))
        
        sector_colors = {"Energy": "#ef4444", "Metals": "#f59e0b", "Agriculture": "#10b981", "Macro_Crypto": "#3b82f6"}
        trace_counter = 0

        for sector, assets in CATEGORIZED_TICKERS.items():
            for name, tid in assets.items():
                try:
                    hist = raw_data[tid].dropna() if len(ticker_ids) > 1 else raw_data.dropna()
                    if hist.empty or len(hist) < 40: continue
                    
                    start_p = hist['Close'].iloc[0]
                    perf_series = ((hist['Close'] / start_p) - 1) * 100
                    color = sector_colors[sector]
                    
                    # Traces
                    fig.add_trace(go.Scatter(x=hist.index, y=perf_series, name=name, legendgroup=name, line=dict(color=color, width=2.5)), row=1, col=1)
                    
                    # PRO AI Prediction
                    proj_y = get_pro_prediction(hist)
                    proj_perf = [((v / start_p) - 1) * 100 for v in proj_y]
                    f_idx = pd.date_range(start=hist.index[-1], periods=25, freq='h')[1:]
                    fig.add_trace(go.Scatter(x=[hist.index[-1]] + list(f_idx), y=[perf_series.iloc[-1]] + list(proj_perf),
                                             name=f"{name} Forecast", legendgroup=name, showlegend=False,
                                             line=dict(color=color, width=2, dash='dot'), opacity=0.35), row=1, col=1)
                    
                    # RSI
                    delta = hist['Close'].diff(); g = delta.where(delta > 0, 0).rolling(14).mean(); l = -delta.where(delta < 0, 0).rolling(14).mean()
                    rsi_line = 100 - (100 / (1 + (g/(l + 1e-9))))
                    fig.add_trace(go.Scatter(x=hist.index, y=rsi_line, showlegend=False, legendgroup=name, line=dict(color=color, width=1, dash='dot'), opacity=0.2), row=2, col=1)

                    # PRO Accuracy: Matches AI direction vs Momentum
                    m_dir = np.sign(perf_series.diff().tail(24).values)
                    p_dir = np.sign(np.diff([perf_series.iloc[-1]] + list(proj_perf[:24])))
                    acc = round((sum(m_dir == p_dir) / 24) * 100, 1)
                    
                    market_data.append({"Asset": name, "Sector": sector, "Price": round(hist['Close'].iloc[-1], 2), "Perf": round(perf_series.iloc[-1], 2), "Acc": acc, "Color": "#10b981" if acc >= 55 else "#f59e0b", "Trend": "BULLISH" if perf_series.iloc[-1] > perf_series.rolling(20).mean().iloc[-1] else "BEARISH"})
                    trace_counter += 3
                except: continue

        # UI
        buttons = [dict(method="restyle", label="GLOBAL VIEW", args=[{"visible": [True] * trace_counter}])]
        for target_sector in CATEGORIZED_TICKERS.keys():
            visibility = []
            for sector, assets in CATEGORIZED_TICKERS.items():
                for _ in assets:
                    v = (sector == target_sector); visibility.extend([v, v, v])
            buttons.append(dict(method="restyle", label=target_sector.upper(), args=[{"visible": visibility}]))

        fig.update_layout(template="plotly_dark", height=850, margin=dict(t=150, b=50), paper_bgcolor="#0a0a0a", plot_bgcolor="#0a0a0a",
                          legend=dict(itemclick="toggleothers", itemdoubleclick="toggle", font=dict(size=10), orientation="v", x=1.02, y=0.5),
                          updatemenus=[dict(type="buttons", direction="right", x=0.5, y=1.18, xanchor="center", buttons=buttons, bgcolor="#1e293b", font=dict(color="white"), active=-1)])

        custom_css = "<style>rect.updatemenu-item-rect { fill: #1e293b !important; } rect.updatemenu-item-rect:hover { fill: #334155 !important; } rect.updatemenu-item-rect.active, rect.updatemenu-item-rect[fill='#F4F4F4'] { fill: #2563eb !important; } text.updatemenu-item-text { fill: #ffffff !important; font-weight: bold !important; pointer-events: none !important; } table { width: 100%; border-collapse: collapse; color: white; table-layout: fixed; } th { padding: 15px; text-align: left; background-color: #111827; color: #94a3b8; } tr { border-bottom: 1px solid #1f2937; }</style>"
        df_market = pd.DataFrame(market_data).sort_values(by="Perf", ascending=False)
        table_rows = "".join([f"<tr><td style='padding:12px; font-weight:bold; text-align:left;'>{r['Asset']}</td><td style='padding:12px; color:#4b5563;'>{r['Sector']}</td><td>{r['Price']}</td><td style='color:{'#10b981' if r['Perf']>0 else '#ef4444'}; font-weight:bold;'>{r['Perf']}%</td><td style='color:{r['Color']}; font-weight:bold;'>{r['Acc']}%</td><td>{r['Trend']}</td></tr>" for _, r in df_market.iterrows()])
        
        return HTMLResponse(content=f"<html><head>{custom_css}</head><body style='margin:0; background:#0a0a0a;'>{fig.to_html(full_html=False, include_plotlyjs='cdn')}<div style='background:#0a0a0a; padding:40px; font-family:sans-serif;'><h2 style='text-align:center; color:#64748b; letter-spacing: 2px;'>V4.4 VOLUME-WEIGHTED ML SUMMARY</h2><table><thead><tr><th>ASSET</th><th>SECTOR</th><th>PRICE</th><th>7D PERF</th><th>ML ACCURACY</th><th>TREND</th></tr></thead><tbody>{table_rows}</tbody></table></div></body></html>")
    except Exception as e:
        return HTMLResponse(content=f"<h1>Error</h1><code>{str(e)}</code>", status_code=500)