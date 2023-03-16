import mlflow
from mlflow.tracking.client import MlflowClient
import os
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf

mlflow.set_tracking_uri('http://localhost:5000')
os.environ['MLFLOW_TRACKING_URI'] = 'postgresql+psycopg2://postgres:password@localhost:5432/mlflowdb'
os.environ['MLFLOW_S3_ENDPOINT_URL'] =  'http://10.180.146.26:9000'
os.environ['AWS_ACCESS_KEY_ID'] = 'admin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'password'

def getRunToUse(metric_to_use):
	client = MlflowClient()
	exp = client.get_experiment_by_name("Alpha")
	runs = mlflow.list_run_infos(exp.experiment_id)
	runid = None
	runidToUse = None
	bestMetricValue = -float('inf')
	for i in runs:
		runid = i.run_id
		run = client.get_run(runid)
		curr_run_metric = run.data.metrics[metric_to_use]
		if curr_run_metric > bestMetricValue:
			bestMetricValue = curr_run_metric
			runidToUse = runid

	return runidToUse


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
	with mlflow.start_run(experiment_id = '0') as r:
		client = MlflowClient()
		runid = getRunToUse('trainAccuracy')
		artifacts = client.list_artifacts(runid)
		print('RunId with best Training Accuracy:{}'.format(runid))
		
		local_dir = './tmp'
		if not os.path.exists(local_dir):
			os.mkdir(local_dir)
		
		tokenizer_path = client.download_artifacts(runid, "tokenizer.pkl", local_dir)
		tokenizer = pickle.load(open(tokenizer_path,'rb'))
		# Loading the tokenizer object
		#tokenizer_path = 'tokenizer.pkl'
		#tokenizer = pickle.load(open(tokenizer_path,'rb'))
			
		# Loading the model
		auri = [a for a in artifacts if a.path != 'tokenizer.pkl']	
		
		artifact_id = auri[0].path.split('_')[1]
		artifact_path = 'runs:/{0}/artifact_{0}'.format(artifact_id)
		keras_model = mlflow.keras.load_model(artifact_path)

		comments, targets = extractor('./')
		#Converting to numpy arrays
		comments = np.array(comments)

		targets = [i-1 for i in targets]
		targets = np.array(targets)
	
		run = client.get_run(runid)
		testSequences = tokenizer.texts_to_sequences(comments)
			
		paddedTestSequences = pad_sequences(testSequences,padding = run.data.params['seq_padding_style'],truncating = run.data.params['seq_truncating_style'],maxlen = int(run.data.params['max_length']))

		targets = tf.keras.utils.to_categorical(targets, 4)
		val_accuracy = keras_model.evaluate(paddedTestSequences, targets, batch_size = int(run.data.params['bs']))[1]
		print('Validation Accuracy is: {}%'.format(val_accuracy))
		print()
		

