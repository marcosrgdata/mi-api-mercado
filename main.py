from fastapi import FastAPI
import yfinance as yf
import pandas as pd
from supabase import create_client, Client
import os

app = FastAPI()

URL_SB = os.getenv("URL_SB")
KEY_SB = os.getenv("KEY_SB")

try:
    supabase: Client = create_client(URL_SB, KEY_SB)
except:
    supabase = None

@app.get("/")
def home():
    return {
        "message": "Industrial Intelligence Platform Active",
        "monitored_assets": ["Gold", "Oil", "Copper", "Silver", "Natural_Gas"],
        "status": "online"
    }

@app.get("/market-intelligence")
def get_market_intelligence():
    tickets = {
        "Gold": "GC=F",
        "Oil": "CL=F",
        "Copper": "HG=F",
        "Silver": "SI=F",
        "Natural_Gas": "NG=F"
    }
    analysis = {}

    for name, ticker_id in tickets.items():
        try:
            ticker = yf.Ticker(ticker_id)
            hist = ticker.history(period="7d")
            if len(hist) < 2: continue

            current_price = round(hist['Close'].iloc[-1], 2)
            weekly_mean = hist['Close'].mean()
            # Traducimos a términos financieros reales: Bullish / Bearish
            trend = "BULLISH" if current_price > weekly_mean else "BEARISH"
            volatility = (hist['Close'].std() / weekly_mean)
            risk = "HIGH" if volatility > 0.02 else "LOW"

            if supabase:
                try:
                    # Seguimos guardando en tu tabla de siempre
                    data_to_save = {"activo": name, "precio": current_price, "tendencia": trend}
                    supabase.table("precios_historicos").insert(data_to_save).execute()
                except: pass

            analysis[name] = {
                "price": current_price,
                "trend_7d": trend,
                "risk_level": risk,
                "recommendation": "WATCH" if risk == "HIGH" else "STABLE"
            }
        except: continue
    
    return analysis

@app.get("/historical-stats")
def get_historical_stats(asset: str = "Gold"):
    if not supabase:
        return {"error": "Database service unavailable"}
    
    try:
        response = supabase.table("precios_historicos")\
            .select("precio")\
            .eq("activo", asset)\
            .order("id", desc=True)\
            .limit(20)\
            .execute()
        
        prices = [row['precio'] for row in response.data]
        
        if len(prices) < 2:
            return {"message": f"Collecting data for {asset}. Try again in a few minutes."}
        
        return {
            "asset": asset,
            "analysis_period": "Last 20 captures",
            "recent_average_price": round(sum(prices) / len(prices), 2),
            "max_detected": max(prices),
            "min_detected": min(prices),
            "volatility": round(pd.Series(prices).std(), 4)
        }
    except Exception as e:
        return {"error": str(e)}