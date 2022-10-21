from mlflow.tracking.client import MlflowClient
import mlflow
from threading import *
import time
import json
import os

def startExperiment(hyperparams, exp_id):
	client = MlflowClient()
	main_exp_id = mlflow.get_experiment_by_name('BaseRayExperiment').experiment_id
	run = client.create_run(main_exp_id)#, tags = {"dataPath":hyperparams["dataPath"]})
	#temp = hyperparams.pop("dataPath", None)
	#with mlflow.start_run(run_id=run.info.run_id, nested = True) as prun:
	#	mlflow.log_params(hyperparams)
	#hyperparams["run_id"] = run.info.run_id
	mlflow.run(run_id = run.info.run_id, uri = '.', entry_point = "ht", use_conda = False, parameters = hyperparams)

def createExperimentAndDataArtifacts(experimentName):
	experiment_id = mlflow.create_experiment(experimentName)#, artifact_location='s3://mlflow')
	return experiment_id


def fetch_hyperparams():
	hyperparams = json.load(open("data.json"))
	return hyperparams


def addExperimentNameToHyperparams(hyperparams, experimentName):
	for hset in hyperparams:
		hset['experiment_name'] = experimentName

	return hyperparams
	
if __name__ == '__main__':
	#mlflow.set_tracking_uri('http://localhost:5000')
	client = MlflowClient()

	# Tracking for the main experiment
	main_experiment_id = mlflow.create_experiment("BaseRayExperiment")

	# This is where model training experiments will be logged
	experimentName = 'RayExperiments'
	exp_id = createExperimentAndDataArtifacts(experimentName)
	hyperparams = fetch_hyperparams()
	hyperparams = addExperimentNameToHyperparams(hyperparams,experimentName)
	
	for hset in hyperparams:	
		startExperiment(hset, exp_id)

