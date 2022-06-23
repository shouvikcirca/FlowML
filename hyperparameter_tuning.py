from mlflow.tracking.client import MlflowClient
import mlflow

def createRun(hyperparams, exp_id):
	client = MlflowClient()
	run = client.create_run(exp_id, tags = {"dataPath":hyperparams["dataPath"]})
	temp = hyperparams.pop("dataPath", None)
	with mlflow.start_run(run_id=run.info.run_id, nested = True) as prun:
		mlflow.log_params(hyperparams)
	mlflow.run(run_id = run.info.run_id, uri = '.', entry_point = "ht", use_conda = False, parameters = hyperparams)


def createExperimentAndDataArtifacts():
	experiment_id = mlflow.create_experiment("HTExperiment")
	return experiment_id


if __name__ == '__main__':
	with mlflow.start_run(experiment_id = '0') as r:
		hyperparams = [
			{
				"learning_rate":0.3,
				"dataPath":'./',
				"vocab_length":100,
				"seq_padding_style":"post",
				"seq_truncating_style":"post",
				"embedding_dim":50,
				"bs":64,
				"epochs":5,
				"max_length":50,
			},
			{
				"learning_rate":0.4,
				"dataPath":'./',
				"vocab_length":100,
				"seq_padding_style":"post",
				"seq_truncating_style":"post",
				"embedding_dim":50,
				"bs":64,
				"epochs":10,
				"max_length":50,
			},
			{
				"learning_rate":0.5,
				"dataPath":'./',
				"vocab_length":100,
				"seq_padding_style":"post",
				"seq_truncating_style":"post",
				"embedding_dim":50,
				"bs":64,
				"epochs":15,
				"max_length":50,
			},
			{
				"learning_rate":0.6,
				"dataPath":'./',
				"vocab_length":100,
				"seq_padding_style":"post",
				"seq_truncating_style":"post",
				"embedding_dim":50,
				"bs":64,
				"epochs":20,
				"max_length":50,
			},
			{
				"learning_rate":0.7,
				"dataPath":'./',
				"vocab_length":100,
				"seq_padding_style":"post",
				"seq_truncating_style":"post",
				"embedding_dim":50,
				"bs":64,
				"epochs":25,
				"max_length":50,
			}
		]

		exp_id = createExperimentAndDataArtifacts()
		for i in range(len(hyperparams)):
				createRun(hyperparams[i], exp_id)

