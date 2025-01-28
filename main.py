from fastapi import FastAPI, UploadFile, Query, HTTPException, Request, File
from fastapi.responses import JSONResponse
import torch
from LungCancer.final_code import GATWithDimensionalityReduction, in_channels, hidden_channels, out_channels, reduce_dim, num_heads
from starlette.responses import RedirectResponse
from LungCancer.engine import run_model
from typing import List
from fastapi.middleware.cors import CORSMiddleware
import overpy
import os
import uvicorn
from WasteClassification.chat import inverse_kinematics, process_image, run_chat_model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

current_dir = os.path.dirname(os.path.abspath(__file__))
lung_cancer_dir = os.path.join(current_dir, "LungCancer")

model = GATWithDimensionalityReduction(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels, reduce_dim=reduce_dim, num_heads=num_heads)
model.load_state_dict(torch.load(os.path.join(lung_cancer_dir, "GATWithDimensionalityReduction.pth"), map_location=torch.device('cpu'), weights_only=True))
model.eval()

@app.post("/predict")
async def predict(files: List[UploadFile]):
    risk = await run_model(files)
    response_data = {"risk": round(risk*100, 2)}

    print(response_data)
    return JSONResponse(content=response_data)

@app.get("/", response_class=RedirectResponse)
async def redirect():
    return RedirectResponse(url="http://localhost:5173/")

api = overpy.Overpass()

@app.get("/search_poi")
async def search_poi(
    q: str = Query(..., description="Query term for the POI, e.g., 'hospital'"),
    lat: float = Query(..., description="Latitude of the search center"),
    lon: float = Query(..., description="Longitude of the search center"),
    radius: int = Query(1000, description="Radius in meters for the search"),
    limit: int = Query(10, description="Limit the number of results")
):
    overpass_query = f"""
    [out:json][timeout:25];
    nwr(around:{radius},{lat},{lon})["amenity"="{q}"];
    out center;
    """
    response = api.query(overpass_query)
    result = []
    for node in response.nodes:
        name = node.tags.get("name", "n/a")
        if(name != "n/a"):
            result.append(name)

    return result[:limit]

@app.post("/process_image")
async def classify(image: UploadFile = File(...)):
    img_str, uploaded_file = await process_image(image)
    classification_result = run_chat_model(img_str, uploaded_file)

    return JSONResponse({
        'classification': classification_result,
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)