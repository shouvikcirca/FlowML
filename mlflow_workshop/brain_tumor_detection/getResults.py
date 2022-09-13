import mlflow
from mlflow.tracking.client import MlflowClient
import os
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf
import requests
import json
import base64
import pandas as pd
from keras import backend as K
from PIL import Image
#import warnings
#warnings.filterwarnings("ignore", category=DeprecationWarning)

mlflow.set_tracking_uri('http://localhost:5000')
#os.environ['MLFLOW_TRACKING_URI'] = 'postgresql+psycopg2://postgres:password@localhost:5432/mlflowdb'
os.environ['MLFLOW_S3_ENDPOINT_URL'] =  'http://10.180.146.26:9000'
os.environ['AWS_ACCESS_KEY_ID'] = 'admin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'password'

if __name__ == "__main__":
	client = MlflowClient()
	#model_name='alphabt'
	stage='Staging'
	model = mlflow.pyfunc.load_model(
			model_uri='models:/alphabt/Staging'
			)

	#print(dir(model) # To get the attributes of the model object
	run_id = model.metadata.run_id
	r = client.get_run(run_id)
	filePath = r.data.tags["dataPath"]
	
	images_yes = os.listdir(filePath+'test/yes')	
	images_no = os.listdir(filePath+'test/no')	

	yes_array = np.empty([len(images_yes),128,128,3])
	for i in range(len(images_yes)):
		y = Image.open(filePath+'test/yes'+'/'+images_yes[i])
		y = y.resize((128,128))
		y = np.asarray(y)
		yes_array[i] = y
	
	no_array = np.empty([len(images_no),128,128,3])
	for i in range(len(images_no)):
		n = Image.open(filePath+'test/no'+'/'+images_no[i])
		n = n.resize((128,128))
		n = np.asarray(n)
		no_array[i] = n
		
	image_array = np.concatenate((yes_array, no_array),axis=0)

	prediction = requests.post(url='http://127.0.0.1:5004/invocations',json={"inputs":image_array.tolist()}, headers={'Content-Type':'application/json',})
	p = prediction.json()
	print(np.array(p).shape)
