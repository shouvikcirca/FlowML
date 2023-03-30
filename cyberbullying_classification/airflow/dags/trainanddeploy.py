import airflow
from airflow.operators.python import PythonOperator
#from airflow.operators.python import BashOperator
from airflow import DAG
import pymongo
from pymongo import MongoClient
import pendulum
import json
from bson import json_util
from mlflow.tracking.client import MlflowClient
import mlflow
import subprocess
import docker

def get_experiments(**context):
	connectionString = 'mongodb://root:password@localhost:27015/mlflowexperiments?authSource=admin'
	client = MongoClient(connectionString)
	db = client['mlflowexperiments']
	collection = db["experimentnames"]
	res = collection.find({"status":"notexecuted"})
	res_list = []
	for r in res:
		res_list.append(json_util.dumps(r))
	
	# XCom push
	with open('unexecuted.json','w') as f:
		json.dump(res_list, f)
	#context['task_instance'].xcom_push(key="unexecuted", value=res_list)
	
def get_experiment_to_run(**context):
	# XCom pull
	#res = context['task_instance'].xcom_pull(task_ids="getUnexecutedExperiments", key="unexecuted")
	unexecutedExperimentNames = []
	with open('unexecuted.json','r') as f:
		data = json.loads(f.read())
		for d in data:
			e = json.loads(d)
			unexecutedExperimentNames.append(e['expname'])
	
	context['task_instance'].xcom_push(key="exptoexecute", value=unexecutedExperimentNames[0])
	#jsonSerialized_unexecutedExperimentNames = json.dumps(unexecutedExperimentNames)
	

def createExperiment(**context):
	mlflow.set_tracking_uri('http://localhost:5000')
	experimentName = context['task_instance'].xcom_pull(task_ids="retrieveExperimentToExecute", key="exptoexecute")
	
	# Create Experiment
	experiment_id = mlflow.create_experiment(experimentName)
	context['task_instance'].xcom_push(key="receivedexptoexecute", value=experimentName)


def getRanges(**context):
	connectionString = 'mongodb://root:password@localhost:27015/mlflowexperiments?authSource=admin'
	client = MongoClient(connectionString)
	db = client['mlflowexperiments']
	collection = db["experimentranges"]

	experimentName = context['task_instance'].xcom_pull(task_ids="retrieveExperimentToExecute", key="exptoexecute")
	res_hp = collection.find({"experiment_name":experimentName})
	hyperparams = ''
	for r in res_hp:
		hyperparams = json_util.dumps(r)
	hyperparams = json.loads(hyperparams)
	popped  = hyperparams.pop("_id", None)
	hyperparams = json.dumps(hyperparams)
	context['task_instance'].xcom_push(key="hparams", value=hyperparams)

def kickoffExperimentation(**context):
	hyperparams = context['task_instance'].xcom_pull(task_ids="getExperimentHpRanges", key="hparams")
	hyperparams = json.loads(hyperparams)
	print('Kickoff Experimentation')
	print(hyperparams)

	_ = mlflow.create_experiment('BaseExperiment')
	client = MlflowClient()
	main_exp_id = mlflow.get_experiment_by_name('BaseExperiment').experiment_id
	run = client.create_run(main_exp_id)#, tags = {"dataPath":hyperparams["dataPath"]})
	mlflow.run(run_id = run.info.run_id, uri = '.', entry_point = "ht", use_conda = False, parameters = hyperparams)

def getModel(**context):
	metric_to_use = 'trainAccuracy'
	experimentName = context['task_instance'].xcom_pull(task_ids="retrieveExperimentToExecute", key="exptoexecute")
	client = MlflowClient()
	exp = client.get_experiment_by_name(experimentName)
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
	
	print('Retrieved best runid')
	context['task_instance'].xcom_push(key="bestRun", value = runidToUse)

def deployBestModel(**context):
	experimentName = context['task_instance'].xcom_pull(task_ids="retrieveExperimentToExecute", key="exptoexecute")
	runid = context['task_instance'].xcom_pull(task_ids="getBestModel", key="bestRun")
	client = MlflowClient()
	model_name = '{}_{}'.format(experimentName, runid)
	client.create_registered_model(model_name)
	mv = client.create_model_version(model_name, source="s3://mlflow/1/{}/artifacts/artifact_{}".format(runid, runid))
	client.transition_model_version_stage(
			name=model_name,
			version=1,
			stage='Staging'
	)

def modelServing(**context):
	print('Initiating Serving')
	experimentName = context['task_instance'].xcom_pull(task_ids="retrieveExperimentToExecute", key="exptoexecute")
	runid = context['task_instance'].xcom_pull(task_ids="getBestModel", key="bestRun")
	model_name = '{}_{}'.format(experimentName, runid)


	cmdString = "nohup mlflow models serve -m 'models:/{}_{}/Staging' -h 127.0.0.1 -p 5004 --env-manager=local &".format(experimentName, runid)
	p2 = subprocess.run(cmdString, shell = True)
	print("Deployment created")


	connectionString = 'mongodb://root:password@localhost:27015/mlflowexperiments?authSource=admin'
	dbClient = MongoClient(connectionString)
	db = dbClient['mlflowexperiments']
	collection = db['deployedModels']
	records = collection.find({})[0]
	collection.update_one({"_id":records['_id']},{'$set':{'cyberbullying_classification': str(runid)}})
	print("Deployed model updated in database")
	
	"""
	cmdString = 'mlflow models serve -m "models:/{}/Staging" -h 127.0.0.1 -p 5004 --env-manager=local'.format(model_name)
	p1 = subprocess.run(cmdString, shell = True, capture_output = True, text = True)
	print(p1.stdout)
	print('Maybe it works')
	"""

dag = DAG(
		dag_id = "trainanddeploy",
		#start_date = airflow.utils.dates.days_ago(3),
		start_date = pendulum.yesterday(),
		schedule_interval=None,
)

getUnexecutedExperiments = PythonOperator(
		task_id = "getUnexecutedExperiments",
		python_callable = get_experiments,
		dag = dag
)

retrieveExperimentToExecute = PythonOperator(
		task_id="retrieveExperimentToExecute",
		python_callable = get_experiment_to_run,
		dag=dag
)

createMlflowExperiment = PythonOperator(
		task_id='createMlflowExperiment',
		python_callable = createExperiment,
		dag = dag
)

getExperimentHpRanges = PythonOperator(
		task_id="getExperimentHpRanges",
		python_callable = getRanges,
		dag=dag
)

kickOffTuning = PythonOperator(
		task_id="kickoffTuning",
		python_callable = kickoffExperimentation,
		dag=dag
)

getBestModel = PythonOperator(
		task_id='getBestModel',
		python_callable = getModel,
		dag=dag
)

registerAndStageModel = PythonOperator(
		task_id='registerAndStageModel',
		python_callable = deployBestModel,
		dag=dag
)

serveModel = PythonOperator(
		task_id='serveModel',
		python_callable = modelServing,
		dag=dag
)

getUnexecutedExperiments >> retrieveExperimentToExecute >> createMlflowExperiment >> getExperimentHpRanges >> kickOffTuning >> getBestModel >> registerAndStageModel >> serveModel
