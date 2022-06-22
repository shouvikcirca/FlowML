#import os
#import numpy as np
#import tempfile
#from mlflow.tracking.fluent import _get_experiment_id
#import argparse
#parser = argparse.ArgumentParser()
#parser.parse_args()
#parser.add_argument("--learningrate")
#args = parser.parse_args()

import mlflow
from mlflow.tracking.client import MlflowClient
import click

@click.command()
@click.option("--learning-rate", default=0.1, type=float)
def runBaseRunner(learning_rate):
	client = MlflowClient()
	with mlflow.start_run() as active_run:
		r = client.get_run(active_run.info.run_id)


		client.log_param(r.info.run_id, "learningRate","Learning rate for this run is {}".format(learning_rate))



if __name__ == "__main__":
	runBaseRunner()







"""
	with mlflow.start_run() as active_run:
	#Extracting comments and targets
		comments, targets = extractor(filePath)

		#Converting to numpy arrays
		comments = np.array(comments)

		targets = [i-1 for i in targets]
		targets = np.array(targets)
		
		localFile = tempfile.mkdtemp()
		Xfile = os.path.join(localFile,'comments.csv')
		yfile = os.path.join(localFile, 'targets.csv')
		mlflow.log_artifact("comments_{}.csv".format(bs), artifact_path="comments.csv")
		mlflow.log_artifact("targets_{}.csv".format(bs), artifact_path="targets.csv")
"""


