from fastapi import FastAPI
import yfinance as yf
import pandas as pd
from supabase import create_client, Client
import os

app = FastAPI()

# Intentamos conectar, si faltan las llaves, no matamos la app todavía
URL_SB = os.getenv("URL_SB")
KEY_SB = os.getenv("KEY_SB")

try:
    supabase: Client = create_client(URL_SB, KEY_SB)
except:
    supabase = None

@app.get("/")
def home():
    return {"mensaje": "API de Inteligencia Industrial Activa", "status": "online"}

@app.get("/inteligencia-mercado")
def obtener_analisis_avanzado():
    tickets = {
        "Oro": "GC=F",
        "Petroleo": "CL=F",
        "Cobre": "HG=F",
        "Plata": "SI=F",
        "Gas_Natural": "NG=F"
    }
    analisis = {}

    for nombre, ticker_id in tickets.items():
        try:
            ticker = yf.Ticker(ticker_id)
            hist = ticker.history(period="7d")
            
            if len(hist) < 2: continue

            precio_actual = round(hist['Close'].iloc[-1], 2)
            media_semanal = hist['Close'].mean()
            tendencia = "ALCISTA" if precio_actual > media_semanal else "BAJISTA"
            volatilidad = (hist['Close'].std() / media_semanal)
            riesgo = "ALTO" if volatilidad > 0.02 else "BAJO"

            # --- ESCUDO PARA LA BASE DE DATOS ---
            if supabase:
                try:
                    data_to_save = {
                        "activo": nombre,
                        "precio": precio_actual,
                        "tendencia": tendencia
                    }
                    supabase.table("precios_historicos").insert(data_to_save).execute()
                except Exception as db_error:
                    print(f"Error al guardar {nombre} en DB: {db_error}")
            # ------------------------------------

            analisis[nombre] = {
                "valor": precio_actual,
                "tendencia_7d": tendencia,
                "nivel_de_riesgo": riesgo,
                "recomendacion": "VIGILAR" if riesgo == "ALTO" else "ESTABLE"
            }
        except Exception as e:
            print(f"Error procesando {nombre}: {e}")
            continue
    
    return analisis