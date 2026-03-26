from fastapi import FastAPI
import yfinance as yf
import pandas as pd
from supabase import create_client, Client
import os

app = FastAPI()

# --- CONFIGURACIÓN DE CONEXIÓN (Variables de Entorno) ---
URL_SB = os.getenv("URL_SB")
KEY_SB = os.getenv("KEY_SB")

try:
    # Intentamos conectar a Supabase, si falla la API no "muere"
    supabase: Client = create_client(URL_SB, KEY_SB)
except Exception as e:
    print(f"Aviso: Supabase no conectado correctamente: {e}")
    supabase = None

@app.get("/")
def home():
    return {
        "mensaje": "Plataforma de Inteligencia Industrial Activa",
        "activos_monitoreados": ["Oro", "Petroleo", "Cobre", "Plata", "Gas_Natural"],
        "status": "online"
    }

@app.get("/inteligencia-mercado")
def obtener_analisis_avanzado():
    """Endpoint principal: Precios en tiempo real y análisis de riesgo"""
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

            # --- ESCUDO DE PERSISTENCIA (Supabase) ---
            if supabase:
                try:
                    data_to_save = {
                        "activo": nombre,
                        "precio": precio_actual,
                        "tendencia": tendencia
                    }
                    supabase.table("precios_historicos").insert(data_to_save).execute()
                except Exception as db_error:
                    print(f"Error guardando {nombre} en DB: {db_error}")

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

@app.get("/estadisticas-historicas")
def obtener_estadisticas(activo: str = "Oro"):
    """Endpoint PRO: Consulta el histórico de tu base de datos y calcula promedios"""
    if not supabase:
        return {"error": "Servicio de base de datos no disponible temporalmente."}
    
    try:
        # Consultamos los últimos 20 registros del activo seleccionado
        response = supabase.table("precios_historicos")\
            .select("precio")\
            .eq("activo", activo)\
            .order("id", desc=True)\
            .limit(20)\
            .execute()
        
        precios = [row['precio'] for row in response.data]
        
        if not precios or len(precios) < 2:
            return {
                "mensaje": f"Recopilando datos para {activo}. Vuelva a intentar en unos minutos.",
                "muestras_actuales": len(precios)
            }
        
        return {
            "activo": activo,
            "analisis_periodo": "Últimas 20 capturas",
            "precio_medio_reciente": round(sum(precios) / len(precios), 2),
            "maximo_detectado": max(precios),
            "minimo_detectado": min(precios),
            "volatilidad_registrada": round(pd.Series(precios).std(), 4)
        }
    except Exception as e:
        return {"error": f"Error al consultar estadísticas: {str(e)}"}