import docker

def modelServing():#(**context):
	#experimentName = context['task_instance'].xcom_pull(task_ids="retrieveExperimentToExecute", key="exptoexecute")
	#runid = context['task_instance'].xcom_pull(task_ids="getBestModel", key="bestRun")
	experimentName = 'Alpha'
	runid = 'fb3bc3c30e08479dbdcc926d527fc8e3i'
	
	model_name = '{}_{}'.format(experimentName, runid)

	docker_client = docker.from_env()
	port = 5004
	
	container = docker_client.containers.run(
		image = 'python:latest',
		name = 'servemodels',
		ports = {
			"5004/tcp":port
			#"9000/tcp":9000,
			#"5432/tcp":5432
		},
		environment = {
			'MLFLOW_TRACKING_URI':'postgresql+psycopg2://postgres:password@localhost:5432/mlflowdb',
			'MLFLOW_S3_ENDPOINT_URL':'http://minio:9000',
			'AWS_ACCESS_KEY_ID':'admin',
			'AWS_SECRET_ACCESS_KEY':'password'
		},
		detach = True,
		command = ['pip install mlflow','mlflow models serve -m "models:/{}/Staging" -h 0.0.0.0 -p 5004 --env-manager=local'.format(model_name)]
	)

if __name__ == '__main__':
	modelServing()
