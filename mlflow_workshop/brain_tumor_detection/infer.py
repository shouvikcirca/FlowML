import mlflow
from mlflow.tracking.client import MlflowClient
import os
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

mlflow.set_tracking_uri('http://localhost:5000')
#os.environ['MLFLOW_TRACKING_URI'] = 'postgresql+psycopg2://postgres:password@localhost:5432/mlflowdb'
os.environ['MLFLOW_S3_ENDPOINT_URL'] =  'http://10.180.146.26:9000'
os.environ['AWS_ACCESS_KEY_ID'] = 'admin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'password'

def getRunToUse(metric_to_use):
	client = MlflowClient()
	exp = client.get_experiment_by_name("BTExperiment")
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


if __name__ == "__main__":
	with mlflow.start_run(experiment_id = '0') as r:
		client = MlflowClient()
		runid = getRunToUse('trainAccuracy')
		artifacts = client.list_artifacts(runid)
		print('RunId with best Training Accuracy:{}'.format(runid))
			
		# Loading the model
		auri = [a for a in artifacts if a.path != 'model_accuracy_graph.png']	
		
		artifact_id = auri[0].path.split('_')[1]
		artifact_path = 'runs:/{0}/artifact_{0}'.format(artifact_id)
		model = mlflow.keras.load_model(artifact_path)


		dataPath = client.get_run(runid).data.tags["dataPath"]	

		DataGen = ImageDataGenerator(
                                   rescale = 1./255,
                                   rotation_range = 20,
                                   validation_split = 0.2
                                   )
		test_data = DataGen.flow_from_directory(
				directory = dataPath,
				target_size = (128,128),
				batch_size = 1,
				class_mode = 'binary'
                                )

		val_accuracy = model.evaluate(test_data)
		print('Validation Accuracy is: {}%'.format(val_accuracy[1]))
		print()
		

