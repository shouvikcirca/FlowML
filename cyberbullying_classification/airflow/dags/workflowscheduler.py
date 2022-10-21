import airflow
from airflow.operators.python import PythonOperator
from airflow import DAG
import pymongo
from pymongo import MongoClient
import pendulum
import json
from bson import json_util
from mlflow.tracking.client import MlflowClient
import mlflow

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

	_ = mlflow.create_experiment(hyperparams['experiment_name'])
	client = MlflowClient()
	main_exp_id = mlflow.get_experiment_by_name('BaseExperiment').experiment_id
	run = client.create_run(main_exp_id)#, tags = {"dataPath":hyperparams["dataPath"]})
	mlflow.run(run_id = run.info.run_id, uri = '.', entry_point = "ht", use_conda = False, parameters = hyperparams)


dag = DAG(
		dag_id = "workflowscheduler",
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

getUnexecutedExperiments >> retrieveExperimentToExecute >> createMlflowExperiment >> getExperimentHpRanges >> kickOffTuning

