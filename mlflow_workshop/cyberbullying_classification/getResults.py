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
#import warnings
#warnings.filterwarnings("ignore", category=DeprecationWarning)



mlflow.set_tracking_uri('http://localhost:5000')
#os.environ['MLFLOW_TRACKING_URI'] = 'postgresql+psycopg2://postgres:password@localhost:5432/mlflowdb'
os.environ['MLFLOW_S3_ENDPOINT_URL'] =  'http://10.180.146.26:9000'
os.environ['AWS_ACCESS_KEY_ID'] = 'admin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'password'

def extractor(filePath):
	files = [i for i in os.listdir(filePath) if '.txt' in i] # Only .txt files contain data
	comments = []
	targets = []
	for f in files:
		with open(filePath+f, 'r') as r:
			s = r.read().split('\n')
			s = [i.split('\t') for i in s]
			comments.extend(s)	

	targets = [int(item[0]) for item in comments if len(item) == 2]	
	comments = [item[1] for item in comments if len(item) == 2]
	return comments, targets

if __name__ == "__main__":
	client = MlflowClient()
	model_name='alpha'
	stage='Staging'
	model = mlflow.pyfunc.load_model(
			model_uri='models:/alpha/Staging'
			)

	#print(dir(model) # To get the attributes of the model object
	run_id = model.metadata.run_id

	local_dir = './tmp'
	if not os.path.exists(local_dir):
		os.mkdir(local_dir)

	tokenizer_path = client.download_artifacts(run_id, "tokenizer.pkl", local_dir)
	tokenizer = pickle.load(open(tokenizer_path,'rb'))

	comments, targets = extractor('./')
	comments = np.array(comments)
	targets = [i-1 for i in targets]
	targets = np.array(targets)
	run = client.get_run(run_id)
	testSequences = tokenizer.texts_to_sequences(comments)
	paddedTestSequences = pad_sequences(testSequences,padding = run.data.params['seq_padding_style'],truncating = run.data.params['seq_truncating_style'],maxlen = int(run.data.params['max_length']))

	params = {}
	prediction = requests.post(url='http://127.0.0.1:5004/invocations',json={"inputs":paddedTestSequences.tolist()}, headers={'Content-Type':'application/json',})
	p = prediction.json()
	print(np.array(p).shape)
