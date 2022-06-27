import mlflow
from mlflow.tracking.client import MlflowClient
import click

@click.command()
@click.option("--metric_to_use", default='accuracy', type=str)
def getRunToUse(metric_to_use):
	client = MlflowClient()
	exp = client.get_experiment_by_name("HTExperiment")
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

	return runid


if __name__ == "__main__":
	with mlflow.start_run(experiment_id = '0') as r:
		runid = getRunToUse()
		auri = list_artifacts(runid)
		print(auri.artifact_path)




