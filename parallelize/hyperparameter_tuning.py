from mlflow.tracking.client import MlflowClient
import mlflow
from threading import *
import time
import json

def createRun(hyperparams, exp_id):
	client = MlflowClient()
	run = client.create_run(exp_id, tags = {"dataPath":hyperparams["dataPath"]})
	temp = hyperparams.pop("dataPath", None)
	with mlflow.start_run(run_id=run.info.run_id, nested = True) as prun:
		mlflow.log_params(hyperparams)
	hyperparams["run_id"] = run.info.run_id
	mlflow.run(run_id = run.info.run_id, uri = '.', entry_point = "ht", use_conda = False, parameters = hyperparams)


def createExperimentAndDataArtifacts():
	experiment_id = mlflow.create_experiment("HTExperiment")
	return experiment_id


if __name__ == '__main__':
	with mlflow.start_run(experiment_id = '0') as r:
		hyperparams = json.load(open("data.json"))
		exp_id = createExperimentAndDataArtifacts()

		start_stamp = time.time()
		for i in range(len(hyperparams)):
			t1 = Thread(target = createRun, args = (hyperparams[i], exp_id))
			t1.start()


		t1.join()
		exec_time = time.time() - start_stamp
		with open("serial_time.txt", 'w') as f:
			f.write('All runs executed in {}'.format(exec_time))

		
