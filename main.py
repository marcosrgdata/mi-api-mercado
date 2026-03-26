from fastapi import FastAPI
import yfinance as yf
import pandas as pd
from supabase import create_client, Client
import os

app = FastAPI()

URL_SB = os.getenv("URL_SB")
KEY_SB = os.getenv("KEY_SB")
supabase: Client = create_client(URL_SB, KEY_SB)

@app.get("/")
def home():
    return {"mensaje": "API de Inteligencia Industrial Activa"}

@app.get("/inteligencia-mercado")
def obtener_analisis_avanzado():
    # Lista de activos a monitorizar (Añadimos Plata y Gas Natural)
    # SI=F es Plata, NG=F es Gas Natural
    tickets = {
        "Oro": "GC=F",
        "Petroleo": "CL=F",
        "Cobre": "HG=F",
        "Plata": "SI=F",
        "Gas_Natural": "NG=F"
    }
    analisis = {}

    for nombre, ticker_id in tickets.items():
        ticker = yf.Ticker(ticker_id)
        hist = ticker.history(period="7d")
        
        if len(hist) < 2: continue

        precio_actual = round(hist['Close'].iloc[-1], 2)
        media_semanal = hist['Close'].mean()
        tendencia = "ALCISTA" if precio_actual > media_semanal else "BAJISTA"
        riesgo = "ALTO" if (hist['Close'].std() / media_semanal) > 0.02 else "BAJO"

        # --- AQUÍ GUARDAMOS EN LA BASE DE DATOS ---
        data_to_save = {
            "activo": nombre,
            "precio": precio_actual,
            "tendencia": tendencia
        }
        supabase.table("precios_historicos").insert(data_to_save).execute()
        # ------------------------------------------

        analisis[nombre] = {
            "valor": precio_actual,
            "tendencia_7d": tendencia,
            "nivel_de_riesgo": riesgo,
            "recomendacion": "VIGILAR" if riesgo == "ALTO" else "ESTABLE"
        }
    
    return analisis