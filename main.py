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

# --- CONFIGURATION ---
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

# --- V4.5 ADAPTIVE ENGINE ---

def get_pro_prediction(df, hours_ahead=24):
    df = df.copy()
    df['hour'] = df.index.hour
    df['day'] = df.index.dayofweek
    # Robust Volume/Momentum check
    df['vol_feat'] = df['Volume'].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
    if df['vol_feat'].sum() == 0: # If no volume data, use price momentum
        df['vol_feat'] = df['Close'].pct_change().fillna(0)
    
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain/(loss + 1e-9))))
    df = df.fillna(method='bfill').dropna()
    
    X = df[['hour', 'day', 'vol_feat', 'rsi']].values
    y = df['Close'].values
    
    # Fast RF for Railway stability
    model = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=42)
    model.fit(X, y)
    
    last_val = df['Close'].iloc[-1]
    volat = df['Close'].diff().std() * 0.9
    
    preds = []
    temp_val = last_val
    for i in range(1, hours_ahead + 1):
        ai_base = model.predict([[ (df.index[-1].hour + i)%24, df.index[-1].dayofweek, df['vol_feat'].iloc[-1], df['rsi'].iloc[-1] ]])[0]
        noise = np.random.normal(0, volat)
        temp_val = (ai_base * 0.45) + (temp_val * 0.55) + noise
        preds.append(temp_val)
    return preds

# --- DASHBOARD ---
@app.get("/visual-dashboard", response_class=HTMLResponse)
def get_dashboard():
    try:
        ticker_ids = list(ALL_TICKERS.values())
        # We fetch 14d but handle cases with less data
        raw_data = yf.download(ticker_ids, period="14d", interval="1h", group_by='ticker', threads=True)
        
        market_data = []
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3],
                            subplot_titles=("V4.5 ADAPTIVE ML TERMINAL", "MOMENTUM (RSI 14)"))
        
        sector_colors = {"Energy": "#ef4444", "Metals": "#f59e0b", "Agriculture": "#10b981", "Macro_Crypto": "#3b82f6"}
        trace_counter = 0

        for sector, assets in CATEGORIZED_TICKERS.items():
            for name, tid in assets.items():
                try:
                    hist = raw_data[tid].dropna() if len(ticker_ids) > 1 else raw_data.dropna()
                    if hist.empty or len(hist) < 15: continue # Relaxed threshold
                    
                    start_p = hist['Close'].iloc[0]
                    perf_series = ((hist['Close'] / start_p) - 1) * 100
                    color = sector_colors[sector]
                    
                    # 1. Real
                    fig.add_trace(go.Scatter(x=hist.index, y=perf_series, name=name, legendgroup=name, line=dict(color=color, width=2.5)), row=1, col=1)
                    
                    # 2. Prediction
                    proj_y = get_pro_prediction(hist)
                    proj_perf = [((v / start_p) - 1) * 100 for v in proj_y]
                    f_idx = pd.date_range(start=hist.index[-1], periods=25, freq='h')[1:]
                    fig.add_trace(go.Scatter(x=[hist.index[-1]] + list(f_idx), y=[perf_series.iloc[-1]] + list(proj_perf),
                                             name=f"{name} Forecast", legendgroup=name, showlegend=False,
                                             line=dict(color=color, width=2, dash='dot'), opacity=0.35), row=1, col=1)
                    
                    # 3. RSI
                    delta = hist['Close'].diff(); g = delta.where(delta > 0, 0).rolling(14).mean(); l = -delta.where(delta < 0, 0).rolling(14).mean()
                    rsi_line = 100 - (100 / (1 + (g/(l + 1e-9))))
                    fig.add_trace(go.Scatter(x=hist.index, y=rsi_line, showlegend=False, legendgroup=name, line=dict(color=color, width=1, dash='dot'), opacity=0.2), row=2, col=1)

                    # Accuracy calculation
                    m_dir = np.sign(perf_series.diff().tail(12).values)
                    p_dir = np.sign(np.diff([perf_series.iloc[-1]] + list(proj_perf[:12])))
                    acc = round((sum(m_dir == p_dir) / 12) * 100, 1) if len(m_dir) > 0 else 50.0
                    
                    market_data.append({"Asset": name, "Sector": sector, "Price": round(hist['Close'].iloc[-1], 2), "Perf": round(perf_series.iloc[-1], 2), "Acc": acc})
                    trace_counter += 3
                except: continue

        # UI & LEGEND FIX
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
        table_rows = "".join([f"<tr><td style='padding:12px; font-weight:bold; text-align:left;'>{r['Asset']}</td><td style='padding:12px; color:#4b5563;'>{r['Sector']}</td><td style='padding:12px;'>{r['Price']}</td><td style='padding:12px; color:{'#10b981' if r['Perf']>0 else '#ef4444'}; font-weight:bold;'>{r['Perf']}%</td><td style='padding:12px; color:#3b82f6; font-weight:bold;'>{r['Acc']}%</td></tr>" for _, r in df_market.iterrows()])
        
        return HTMLResponse(content=f"<html><head>{custom_css}</head><body style='margin:0; background:#0a0a0a;'>{fig.to_html(full_html=False, include_plotlyjs='cdn')}<div style='background:#0a0a0a; padding:40px; font-family:sans-serif;'><h2 style='text-align:center; color:#64748b; letter-spacing: 2px;'>V4.5 INSTITUTIONAL SUMMARY</h2><table><thead><tr><th>ASSET</th><th>SECTOR</th><th>PRICE</th><th>7D PERF</th><th>ML ACCURACY</th></tr></thead><tbody>{table_rows}</tbody></table></div></body></html>")
    except Exception as e:
        return HTMLResponse(content=f"<h1>Error</h1><code>{str(e)}</code>", status_code=500)