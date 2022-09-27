from fastapi import FastAPI, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get('/', response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("createexperiment.html", {"request": request})


class ExperimentDataModel(BaseModel):
    experimentname: str
    experimenttype: str

@app.post('/experimentData')
async def index(data: ExperimentDataModel):

    hyperparams = {
        'Text Classification':[
            ['learning_rate', 'float'],
            # ['learning_rate_end', 'float'],
            ['vocab_length','int'],
            # ['vocab_length_end','int'],
            ['seq_padding_style','string',['post', 'pre']],
            #['seq_truncating_style','string',['post', 'pre']],
            ['embedding_dim','int'],
            # ['embedding_dim_end','int'],
            ['bs','int'],
            # ['bs_end','int'],
            ['epochs','int'],
            # ['epochs_end','int']
            #['max_length','int']
        ]
    }

    return hyperparams[data.experimenttype]

    # print('{}_{}'.format(data.experimentname, data.experimenttype))



@app.post('/uploadHParams')
async def index(data):
    print(data)
    return {'data':'ok'}
