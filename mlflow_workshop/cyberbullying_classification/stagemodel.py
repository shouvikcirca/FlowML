import mlflow
from mlflow.tracking.client import MlflowClient
import os
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf


mlflow.set_tracking_uri('http://localhost:5000')
#os.environ['MLFLOW_TRACKING_URI'] = 'postgresql+psycopg2://postgres:password@localhost:5432/mlflowdb'
os.environ['MLFLOW_S3_ENDPOINT_URL'] =  'http://10.180.146.26:9000'
os.environ['AWS_ACCESS_KEY_ID'] = 'admin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'password'


if __name__ == "__main__":
	client = MlflowClient()
	#rmodel = client.create_registered_model("alpha")
	#rmodel = client.get_registered_model('alpha')
	#mv = client.create_model_version('alpha')
	
	client.transition_model_version_stage(
			name='alpha',
			version=1,
			stage='Staging'
			)
	
	#client.delete_model_version(name='alpha',version=1)
