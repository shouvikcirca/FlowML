import airflow
from airflow.operators.python import PythonOperator
from airflow.operators.python import BranchPythonOperator
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

def checkRetrainingThreshold(**context):
	connectionString = 'mongodb://root:password@localhost:27015/mlflowexperiments?authSource=admin'
	client = MongoClient(connectionString)
	db = client['mlflowexperiments']
	collection = db["dataCounts"]
	
	existingData = collection.find({"trainingData": {'$exists': True}})[0]
	countExistingData = int(existingData['trainingData'])
	experimentName = existingData['experimentName']

	collection = db["trainingdata"]
	newData = len(collection.find({"comments": {'$exists': True}})[0]['comments'])

	# Checking if new samples have been added to training set
	if newData == countExistingData:
		return "flowexit"
	elif newData > countExistingData:
		context['task_instance'].xcom_push(key="experimentName", value=experimentName)
		return "initiateTraining"
	else:
		return "counterror"


def countErrorFlow(**context):
	print("Error: New Training set count less than existing data count")

def exitFlow(**context):
	print("Not enough samples for retraining")

def trainAgain(**context):
	# Connected to database
	connectionString = 'mongodb://root:password@localhost:27015/mlflowexperiments?authSource=admin'
	dbClient = MongoClient(connectionString)
	db = dbClient['mlflowexperiments']

	# Seeking hyperparameters for experiment. Work on this.
	collection = db['experimentranges']
	experimentName = context['task_instance'].xcom_pull(task_ids="retrainornot", key="experimentName")
	hyperparams = collection.find({"experiment_name":experimentName})[0]
	popped  = hyperparams.pop("_id", None)

	main_exp_id = mlflow.get_experiment_by_name('BaseExperiment').experiment_id
	
	runid = mlflow.list_run_infos(main_exp_id)[0].run_uuid
	mlflow.run(run_id = runid, uri = '.', entry_point = "ht", use_conda = False, parameters = hyperparams)

def getModel(**context):
	metric_to_use = 'trainAccuracy'
	experimentName = context['task_instance'].xcom_pull(task_ids="retrainornot", key="experimentName")
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

def bestmodelchanged(**context):
	bestModel = context['task_instance'].xcom_pull(task_ids="getbestmodel", key="bestRun")
	experimentName = context['task_instance'].xcom_pull(task_ids="retrainornot", key="experimentName")
	
	connectionString = 'mongodb://root:password@localhost:27015/mlflowexperiments?authSource=admin'
	dbClient = MongoClient(connectionString)
	db = dbClient['mlflowexperiments']
	collection = db['deployedModels']
	
	fetchedObject = collection.find({"experimentname": "Alpha"})[0]
	previousBestRun = fetchedObject[fetchedObject['task']]

	if  previousBestRun != bestModel:
		return "deploynewmodel"
	return "samemodelremains"

def deploynew(**context):
	connectionString = 'mongodb://root:password@localhost:27015/mlflowexperiments?authSource=admin'
	dbClient = MongoClient(connectionString)
	db = dbClient['mlflowexperiments']
	collection = db['deployedModels']
	
	bestModel = context['task_instance'].xcom_pull(task_ids="getbestmodel", key="bestRun")
	experimentName = context['task_instance'].xcom_pull(task_ids="retrainornot", key="experimentName")

	# Killing Deployment
	p1 = subprocess.run("sudo fuser -k 5004/tcp", shell = True, capture_output = True, text = True)
	print("Deployment terminated")

	# Deploy new 
	cmdString = "nohup mlflow models serve -m 'models:/{}_{}/Staging' -h 127.0.0.1 -p 5004 --env-manager=local &".format(experimentName, bestModel)

	p2 = subprocess.run(cmdString, shell = True)
	print("Deployment recreated")

	# Change deployed model name in database
	collection = db['deployedModels']
	records = collection.find({})[0]
	collection.update_one({"_id":records['_id']},{'$set':{'cyberbullying_classification': bestModel}})
	print("Deployed model updated in database")


def sameremains(**context):
	print("Previous best model is still the best")


dag = DAG(
		dag_id = "trainonnewdata",
		#start_date = airflow.utils.dates.days_ago(3),
		start_date = pendulum.yesterday(),
		schedule_interval=None,
)

retrainornot = BranchPythonOperator(
		task_id = "retrainornot",
		python_callable = checkRetrainingThreshold,
		dag = dag
)

flowexit = PythonOperator(
		task_id = "flowexit",
		python_callable = exitFlow,
		dag = dag
)

counterror = PythonOperator(
		task_id = "counterror",
		python_callable = countErrorFlow,
		dag = dag
)

initiateTraining = PythonOperator(
		task_id = "initiateTraining",
		python_callable = trainAgain,
		dag = dag
)

getbestmodel = PythonOperator(
		task_id = "getbestmodel",
		python_callable = getModel,
		dag = dag
)

redeploy = PythonOperator(
		task_id = "redeploy",
		python_callable = bestmodelchanged,
		dag = dag
)

deploynewmodel = PythonOperator(
		task_id = "deploynewmodel",
		python_callable = deploynew,
		dag = dag
)

samemodelremains = PythonOperator(
		task_id = "samemodelremains",
		python_callable = sameremains,
		dag = dag
)

retrainornot >> [flowexit, initiateTraining, counterror]
initiateTraining >> getbestmodel >> redeploy
redeploy  >> [deploynewmodel, samemodelremains]


