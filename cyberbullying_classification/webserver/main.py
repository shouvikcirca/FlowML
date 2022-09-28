from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Union
import pymongo
from pymongo import MongoClient
import json
import shutil

app = FastAPI()


connectionString = 'mongodb://root:password@localhost:27015/mlflowexperiments?authSource=admin'
client = MongoClient(connectionString)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get('/', response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("createexperiment.html", {"request": request})


class ExperimentDataModel(BaseModel):
    experimentname: str
    experimenttype: str

class MLModel(BaseModel):
    experimentname: str
    learning_rate_start: str
    learning_rate_end: str
    vocab_length_start: str
    vocab_length_end: str
    embedding_dim_start: str
    embedding_dim_end: str
    bs_start:str
    bs_end:str
    epochs_start:str
    epochs_end:str
    seq_padding_style:str
    # liason: Union[float, None] = None


@app.post('/experimentData')
async def index(data: ExperimentDataModel):

    db = client['mlflowexperiments']

    collection = db["experimentnames"]
    collection.insert_one({'expname':data.experimentname, 'exptype':data.experimenttype})

    collection = db["taskhyperparams"]
    hyperparams = collection.find({'name':'Text Classification'})[0]['hp']
   
    return hyperparams



@app.post('/uploadHParams')
async def index(data:MLModel):
    jsonData = json.loads(data.json())
    db = client['mlflowexperiments']
    collection = db["experimentranges"]
    collection.insert_one(jsonData)


@app.post("/datasetUpload")
async def datasetUpload(file: UploadFile = File(...)):
    save_location = 'datafiles/{}'.format(file.filename)
    with open(save_location,'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)