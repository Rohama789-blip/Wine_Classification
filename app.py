
from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "wine_model.pkl"
TEMPLATE_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="Wine Classification API",
    description="Machine Learning API for Wine Classification",
    version="1.0.0"
)


# =========================================================
# STATIC FILES
# =========================================================

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static"
)


# =========================================================
# TEMPLATES
# =========================================================

templates = Jinja2Templates(directory=TEMPLATE_DIR)


# =========================================================
# LOAD MODEL
# =========================================================

try:
    model = joblib.load(MODEL_PATH)
    print("Wine classification model loaded successfully.")

except Exception as e:
    model = None
    print(f"Model loading error: {e}")


# =========================================================
# HOME PAGE
# =========================================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "prediction": None,
            "error": None
        }
    )


# =========================================================
# PREDICTION
# =========================================================

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request
):

    try:

        form = await request.form()

        # Get values from HTML form
        alcohol = float(form.get("alcohol"))
        malic_acid = float(form.get("malic_acid"))
        ash = float(form.get("ash"))
        alcalinity_of_ash = float(form.get("alcalinity_of_ash"))
        magnesium = float(form.get("magnesium"))
        total_phenols = float(form.get("total_phenols"))
        flavanoids = float(form.get("flavanoids"))
        nonflavanoid_phenols = float(form.get("nonflavanoid_phenols"))
        proanthocyanins = float(form.get("proanthocyanins"))
        color_intensity = float(form.get("color_intensity"))
        hue = float(form.get("hue"))
        od280_od315 = float(form.get("od280_od315"))
        proline = float(form.get("proline"))

        # Create DataFrame
        input_data = pd.DataFrame([{
            "alcohol": alcohol,
            "malic_acid": malic_acid,
            "ash": ash,
            "alcalinity_of_ash": alcalinity_of_ash,
            "magnesium": magnesium,
            "total_phenols": total_phenols,
            "flavanoids": flavanoids,
            "nonflavanoid_phenols": nonflavanoid_phenols,
            "proanthocyanins": proanthocyanins,
            "color_intensity": color_intensity,
            "hue": hue,
            "od280/od315": od280_od315,
            "proline": proline
        }])

        # Check model
        if model is None:
            raise Exception("Model could not be loaded.")

        # Prediction
        prediction = model.predict(input_data)[0]

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "prediction": prediction,
                "error": None
            }
        )

    except Exception as e:

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "prediction": None,
                "error": str(e)
            }
        )


# =========================================================
# API HEALTH CHECK
# =========================================================

@app.get("/api")
async def api_status():

    return {
        "message": "Wine Classification API is running successfully"
    }
