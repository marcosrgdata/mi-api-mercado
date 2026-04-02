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
from sklearn.linear_model import LinearRegression

app = FastAPI()

# --- ASSET LIST ---
CATEGORIZED_TICKERS = {
    "Energy": {"Crude_Oil": "CL=F", "Brent_Oil": "BZ=F", "Natural_Gas": "NG=F", "Gasoline": "RB=F", "Heating_Oil": "HO=F", "Uranium": "URA"},
    "Metals": {"Copper": "HG=F", "Aluminum": "ALI=F", "Lithium": "LIT", "Steel": "SLX", "Gold": "GC=F", "Silver": "SI=F", "Platinum": "PL=F", "Palladium": "PA=F"},
    "Agriculture": {"Corn": "ZC=F", "Wheat": "ZW=F", "Soybeans": "ZS=F", "Coffee": "KC=F", "Sugar": "SB=F", "Cocoa": "CC=F", "Cotton": "CT=F"},
    "Macro_Crypto": {"SP500": "^GSPC", "VIX_Index": "^VIX", "Bitcoin": "BTC-USD", "Carbon_Credits": "KRBN"}
}
ALL_TICKERS = {k: v for cat in CATEGORIZED_TICKERS.values() for k, v in cat.items()}

# --- STOCHASTIC ENGINE V3.20 (THE "REALISTIC" ONE) ---
def get_stochastic_projection(data_series, hours_ahead=24):
    """Generates a realistic forecast with volatility (Random Walk)."""
    window = data_series.tail(30)
    y = window.values.reshape(-1, 1)
    X = np.arange(len(y)).reshape(-1, 1)
    
    # 1. Get the Trend (Drift)
    model = LinearRegression()
    model.fit(X, y)
    drift = model.coef_[0][0]
    
    # 2. Get the Volatility (Noise level)
    # We calculate the standard deviation of changes
    volatility = data_series.diff().std() * 0.8 
    
    # 3. Generate Random Walk
    last_val = data_series.iloc[-1]
    projection = []
    current_val = last_val
    
    for i in range(1, hours_ahead + 1):
        # Trend + Random Noise (Gaussian)
        # We apply a small 'mean reversion' to keep it from going to infinity
        noise = np.random.normal(0, volatility)
        current_val = current_val + drift + noise
        projection.append(current_price_logic(current_val, last_val, i))
        
    return projection

def current_price_logic(val, last, step):
    # Damping factor to avoid unrealistic exponential growth
    decay = 0.99 ** step
    return last + (val - last) * decay

# --- DASHBOARD ---
@app.get("/visual-dashboard", response_class=HTMLResponse)
def get_dashboard():
    try:
        ticker_ids = list(ALL_TICKERS.values())
        raw_data = yf.download(ticker_ids, period="7d", interval="1h", group_by='ticker', threads=True)
        
        market_data = []
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3],
                            subplot_titles=("MARKET PERFORMANCE & STOCHASTIC FORECAST", "MOMENTUM (RSI)"))
        
        sector_colors = {"Energy": "#ef4444", "Metals": "#f59e0b", "Agriculture": "#10b981", "Macro_Crypto": "#3b82f6"}
        trace_counter = 0

        for sector, assets in CATEGORIZED_TICKERS.items():
            for name, tid in assets.items():
                try:
                    hist = raw_data[tid].dropna() if len(ticker_ids) > 1 else raw_data.dropna()
                    if hist.empty or len(hist) < 30: continue
                    
                    perf_series = ((hist['Close'] / hist['Close'].iloc[0]) - 1) * 100
                    color = sector_colors[sector]
                    
                    # Real Line
                    fig.add_trace(go.Scatter(x=hist.index, y=perf_series, name=name, legendgroup=name,
                                             line=dict(color=color, width=2.5)), row=1, col=1)

                    # Stochastic Projection Line (The "Wiggly" line)
                    proj_y = get_stochastic_projection(perf_series, hours_ahead=24)
                    future_index = pd.date_range(start=hist.index[-1], periods=25, freq='h')[1:]
                    
                    fig.add_trace(go.Scatter(x=[hist.index[-1]] + list(future_index), y=[perf_series.iloc[-1]] + list(proj_y),
                                             name=f"{name} Forecast", legendgroup=name, showlegend=False,
                                             line=dict(color=color, width=2, dash='dot'), opacity=0.4), row=1, col=1)
                    
                    # RSI
                    delta = hist['Close'].diff()
                    gain = delta.where(delta > 0, 0).rolling(14).mean(); loss = -delta.where(delta < 0, 0).rolling(14).mean()
                    rsi = 100 - (100 / (1 + (gain/(loss + 1e-9))))
                    fig.add_trace(go.Scatter(x=hist.index, y=rsi, showlegend=False, legendgroup=name,
                                             line=dict(color=color, width=1, dash='dot'), opacity=0.2), row=2, col=1)

                    market_data.append({"Asset": name, "Sector": sector, "Price": round(hist['Close'].iloc[-1], 2), "Perf": round(perf_series.iloc[-1], 2)})
                    trace_counter += 3
                except: continue

        # UI & TABLE (CSS remains the same for stability)
        buttons = [dict(method="restyle", label="GLOBAL VIEW", args=[{"visible": [True] * trace_counter}])]
        for target_sector in CATEGORIZED_TICKERS.keys():
            visibility = []
            for sector, assets in CATEGORIZED_TICKERS.items():
                for _ in assets:
                    v = (sector == target_sector); visibility.extend([v, v, v])
            buttons.append(dict(method="restyle", label=target_sector.upper(), args=[{"visible": visibility}]))

        fig.update_layout(template="plotly_dark", height=850, margin=dict(t=150, b=50), paper_bgcolor="#0a0a0a", plot_bgcolor="#0a0a0a",
                          updatemenus=[dict(type="buttons", direction="right", x=0.5, y=1.18, xanchor="center", buttons=buttons, bgcolor="#1e293b", font=dict(color="white"))])

        custom_css = "<style>rect.updatemenu-item-rect { fill: #1e293b !important; } rect.updatemenu-item-rect:hover { fill: #334155 !important; } rect.updatemenu-item-rect.active, rect.updatemenu-item-rect[fill='#F4F4F4'] { fill: #2563eb !important; } text.updatemenu-item-text { fill: #ffffff !important; font-weight: bold !important; pointer-events: none !important; } table { width: 100%; border-collapse: collapse; color: white; table-layout: fixed; } th { padding: 15px; text-align: left; background-color: #111827; color: #94a3b8; } tr { border-bottom: 1px solid #1f2937; }</style>"
        df_market = pd.DataFrame(market_data).sort_values(by="Perf", ascending=False)
        table_rows = "".join([f"<tr><td style='padding:12px; font-weight:bold;'>{r['Asset']}</td><td style='color:#4b5563;'>{r['Sector']}</td><td>{r['Price']}</td><td style='color:{'#10b981' if r['Perf']>0 else '#ef4444'}; font-weight:bold;'>{r['Perf']}%</td></tr>" for _, r in df_market.iterrows()])
        table_html = f"<div style='background:#0a0a0a; padding:40px; font-family:sans-serif;'><h2 style='text-align:center; color:#64748b;'>MARKET SUMMARY</h2><table><thead><tr><th>ASSET</th><th>SECTOR</th><th>PRICE</th><th>7D PERF</th></tr></thead><tbody>{table_rows}</tbody></table></div>"
        
        return HTMLResponse(content=f"<html><head>{custom_css}</head><body style='margin:0; background:#0a0a0a;'>{fig.to_html(full_html=False, include_plotlyjs='cdn')}{table_html}</body></html>")
    except Exception as e:
        return HTMLResponse(content=f"<h1>Error</h1><code>{str(e)}</code>", status_code=500)