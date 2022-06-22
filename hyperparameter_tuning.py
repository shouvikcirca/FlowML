from mlflow.tracking.client import MlflowClient
import mlflow

def createRun(exp_id, learning_rate):
	params = {
			"learning_rate": learning_rate,
		 }

	run = client.create_run(exp_id)
	mlflow.run(run_id = run.info.run_id, uri = '.', entry_point = "ht", use_conda = False, parameters = params)

if __name__ == '__main__':
	client = MlflowClient()
	for i in [0.1, 0.2, 0.3, 0.4, 0.5]:
		createRun('2', i)

