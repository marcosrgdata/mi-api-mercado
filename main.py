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
from gradio_client import Client

app = FastAPI()

# --- ASSET LIST ---
CATEGORIZED_TICKERS = {
    "Energy": {"Crude_Oil": "CL=F", "Brent_Oil": "BZ=F", "Natural_Gas": "NG=F", "Gasoline": "RB=F", "Heating_Oil": "HO=F", "Uranium": "URA"},
    "Metals": {"Copper": "HG=F", "Aluminum": "ALI=F", "Lithium": "LIT", "Steel": "SLX", "Gold": "GC=F", "Silver": "SI=F", "Platinum": "PL=F", "Palladium": "PA=F"},
    "Agriculture": {"Corn": "ZC=F", "Wheat": "ZW=F", "Soybeans": "ZS=F", "Coffee": "KC=F", "Sugar": "SB=F", "Cocoa": "CC=F", "Cotton": "CT=F"},
    "Macro_Crypto": {"SP500": "^GSPC", "VIX_Index": "^VIX", "Bitcoin": "BTC-USD", "Carbon_Credits": "KRBN"}
}
ALL_TICKERS = {k: v for cat in CATEGORIZED_TICKERS.values() for k, v in cat.items()}

# --- AI CONNECTORS ---
def get_ai_prediction_v5(asset_name, prices_list):
    try:
        if len(prices_list) < 48: return None
        client = Client("marcosrgdata/trading-brain-v5")
        prices_string = ",".join(map(str, prices_list[-48:]))
        return client.predict(asset_name=asset_name, prices_string=prices_string, api_name="/predict_v5")
    except: return None

# --- DASHBOARD ---
@app.get("/visual-dashboard", response_class=HTMLResponse)
def get_dashboard():
    try:
        ticker_ids = list(ALL_TICKERS.values())
        raw_data = yf.download(ticker_ids, period="14d", interval="1h", group_by='ticker', threads=True)
        
        market_data = []
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.75, 0.25],
                            subplot_titles=("V5.0 PRO QUANT TERMINAL", "MOMENTUM (RSI 14)"))
        
        colors = {"Energy": "#ef4444", "Metals": "#f59e0b", "Agriculture": "#10b981", "Macro_Crypto": "#3b82f6"}
        trace_idx = 0

        for sector, assets in CATEGORIZED_TICKERS.items():
            for name, tid in assets.items():
                try:
                    hist = raw_data[tid].dropna() if len(ticker_ids) > 1 else raw_data.dropna()
                    if len(hist) < 48: continue 
                    
                    base_p = hist['Close'].iloc[0] # For 7D Performance
                    perf = round(((hist['Close'].iloc[-1] / base_p) - 1) * 100, 2)
                    
                    # 1. Candlestick Trace
                    fig.add_trace(go.Candlestick(
                        x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'],
                        name=name, legendgroup=name,
                        increasing_line_color=colors[sector], decreasing_line_color='#ffffff'
                    ), row=1, col=1)
                    
                    # 2. RSI Trace
                    delta = hist['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                    rsi = 100 - (100 / (1 + (gain / (loss + 1e-9))))
                    fig.add_trace(go.Scatter(x=hist.index, y=rsi, showlegend=False, legendgroup=name, 
                                             line=dict(color=colors[sector], width=1), opacity=0.3), row=2, col=1)

                    # Data for Table
                    real_24h_max = round(hist['High'].tail(24).max(), 2)
                    real_24h_min = round(hist['Low'].tail(24).min(), 2)
                    ai_v5 = get_ai_prediction_v5(name, hist['Close'].tolist())
                    
                    market_data.append({
                        "Asset": name, "Sector": sector, "Price": round(hist['Close'].iloc[-1], 2), "Perf": perf,
                        "Real_Max": real_24h_max, "Real_Min": real_24h_min,
                        "AI_Trend": ai_v5['prediction'] if ai_v5 else "N/A",
                        "AI_Conf": ai_v5['confidence'] if ai_v5 else "N/A",
                        "AI_Max": ai_v5['expected_max'] if ai_v5 else 0,
                        "AI_Min": ai_v5['expected_min'] if ai_v5 else 0
                    })
                    trace_idx += 2
                except: continue

        # UI BUTTONS & LEGEND (Back to V4.6 Style)
        btns = [dict(method="restyle", label="GLOBAL VIEW", args=[{"visible": [True] * trace_idx}])]
        for s_name in CATEGORIZED_TICKERS.keys():
            vis = []
            for s, a in CATEGORIZED_TICKERS.items():
                for _ in a:
                    v = (s == s_name); vis.extend([v, v])
            btns.append(dict(method="restyle", label=s_name.upper(), args=[{"visible": vis}]))

        fig.update_layout(
            template="plotly_dark", height=900, margin=dict(t=120, b=50), paper_bgcolor="#0a0a0a", plot_bgcolor="#0a0a0a",
            xaxis_rangeslider_visible=False,
            legend=dict(itemclick="toggleothers", itemdoubleclick="toggle", font=dict(size=10), orientation="v", x=1.02, y=0.5),
            updatemenus=[dict(type="buttons", direction="right", x=0.5, y=1.12, xanchor="center", 
                              buttons=btns, bgcolor="#1e293b", font=dict(color="white"))]
        )

        # CSS STYLING (The Button Fix)
        css = """
        <style>
            body { background-color: #0a0a0a; color: #e2e8f0; font-family: sans-serif; margin: 0; }
            rect.updatemenu-item-rect { fill: #1e293b !important; } 
            rect.updatemenu-item-rect:hover { fill: #334155 !important; } 
            rect.updatemenu-item-rect.active { fill: #2563eb !important; } 
            text.updatemenu-item-text { fill: #ffffff !important; font-weight: bold !important; } 
            .main-container { padding: 40px; }
            table { width: 100%; border-collapse: collapse; background: #0a0a0a; table-layout: fixed; }
            th { background: #111827; color: #94a3b8; padding: 15px; text-align: left; font-size: 0.8em; }
            td { padding: 12px 15px; border-bottom: 1px solid #1f2937; font-size: 0.9em; }
            .up { color: #10b981; font-weight: bold; } .down { color: #ef4444; font-weight: bold; }
            .ai-box { border-left: 2px solid #2563eb; padding-left: 10px; }
            .conf { color: #3b82f6; font-size: 0.85em; }
        </style>
        """
        
        # TABLE GENERATION (Including 7D Perf)
        df_m = pd.DataFrame(market_data).sort_values(by="Perf", ascending=False)
        rows = ""
        for _, r in df_m.iterrows():
            t_class = "up" if r['AI_Trend'] == "UP" else "down"
            p_class = "up" if r['Perf'] > 0 else "down"
            rows += f"""
            <tr>
                <td><b>{r['Asset']}</b><br><small style='color:#4b5563;'>{r['Sector']}</small></td>
                <td>${r['Price']}</td>
                <td class="{p_class}">{r['Perf']}%</td>
                <td style='color:#64748b'>MAX: ${r['Real_Max']}<br>MIN: ${r['Real_Min']}</td>
                <td class="ai-box {t_class}">{r['AI_Trend']} <span class="conf">({r['AI_Conf']})</span></td>
                <td class="ai-box">
                    <span class="up">H: ${r['AI_Max']}</span><br>
                    <span class="down">L: ${r['AI_Min']}</span>
                </td>
            </tr>
            """
        
        return HTMLResponse(content=f"""
            <html><head>{css}</head><body>
                {fig.to_html(full_html=False, include_plotlyjs='cdn')}
                <div class="main-container">
                    <h2 style='color:#64748b; letter-spacing:2px;'>V5.0 INSTITUTIONAL SUMMARY</h2>
                    <table>
                        <thead>
                            <tr>
                                <th>ASSET</th><th>PRICE</th><th>7D PERF</th><th>REAL 24H RANGE</th>
                                <th>AI SIGNAL</th><th>AI TARGET (24H)</th>
                            </tr>
                        </thead>
                        <tbody>{rows}</tbody>
                    </table>
                </div>
            </body></html>
        """)
    except Exception as e:
        return HTMLResponse(content=f"<h1>Error</h1><code>{str(e)}</code>", status_code=500)