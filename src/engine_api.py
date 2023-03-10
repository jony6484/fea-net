import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
import torch
from consts import *


app = FastAPI()

# Class for typing
class ExpData(BaseModel):
    heights: List[float]
    pressures: List[float]
    
    
###### The Endpoint for exp #######
@app.post("/post-exp")
async def post_exp(exp_arr: ExpData):
    h_exp = np.array(exp_arr.heights)
    p_exp = np.array(exp_arr.pressures)
    return {'h_max': max(h_exp), 'p_max': max(p_exp)}



###### The Endpoint for checkpoint #######
@app.post("/post-exp")
async def post_exp(file_name):
    model_path = MODEL_PATH / file_name

    return {'path': str(model_path)}