from fastapi import FastAPI
import yfinance as yf
import pandas as pd
from supabase import create_client, Client
from sklearn.linear_model import LinearRegression
import numpy as np
import os
import datetime

app = FastAPI()

# --- SUPABASE SETUP ---
URL_SB = os.getenv("URL_SB")
KEY_SB = os.getenv("KEY_SB")
try:
    supabase: Client = create_client(URL_SB, KEY_SB)
except:
    supabase = None

# --- AI ENGINE ---
def calculate_forecast(prices_list):
    """Calculates a 24h prediction using Linear Regression."""
    y = np.array(prices_list)
    X = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, y)
    
    # R2 Score (Confidence)
    confidence = round(model.score(X, y), 4)
    # Predict next point
    next_index = np.array([[len(y)]])
    prediction = round(model.predict(next_index)[0], 2)
    
    return prediction, confidence

# --- ENDPOINTS ---
@app.get("/")
def home():
    return {"message": "AI-Powered Industrial Intelligence Platform Active", "status": "online"}

@app.get("/market-intelligence")
def get_market_intelligence():
    tickets = {
        "Gold": "GC=F", "Oil": "CL=F", "Copper": "HG=F", 
        "Silver": "SI=F", "Natural_Gas": "NG=F", 
        "Lithium": "LIT", "Uranium": "URA" 
    }
    analysis = {}
    for name, ticker_id in tickets.items():
        try:
            ticker = yf.Ticker(ticker_id)
            hist = ticker.history(period="7d")
            if len(hist) < 2: continue
            
            price = round(hist['Close'].iloc[-1], 2)
            trend = "BULLISH" if price > hist['Close'].mean() else "BEARISH"
            
            if supabase:
                try:
                    supabase.table("precios_historicos").insert({"activo": name, "precio": price, "tendencia": trend}).execute()
                except: pass

            analysis[name] = {"price": price, "trend": trend, "risk": "HIGH" if (hist['Close'].std()/price) > 0.02 else "LOW"}
        except: continue
    return analysis

@app.get("/premium-forecast")
def get_premium_forecast(asset: str = "Gold"):
    """ULTRA ENDPOINT: AI Prediction for the next 24h."""
    if not supabase: return {"error": "DB not connected"}
    
    try:
        response = supabase.table("precios_historicos").select("precio").eq("activo", asset).order("id", desc=True).limit(30).execute()
        prices = [row['precio'] for row in response.data]
        prices.reverse()

        if len(prices) < 15:
            return {"message": f"Need more data for {asset} (Current: {len(prices)}/15)"}

        prediction, r2 = calculate_forecast(prices)
        current = prices[-1]
        
        return {
            "asset": asset,
            "current_price": current,
            "forecast_24h": prediction,
            "confidence_r2": r2,
            "expected_move": "UPWARD" if prediction > current else "DOWNWARD",
            "disclaimer": "AI model based on trends. Not financial advice."
        }
    except Exception as e:
        return {"error": str(e)}