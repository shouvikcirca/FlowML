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

def getRun(**context):
	metric_to_use = 'trainAccuracy'
	experimentName = 'Alpha'
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


def getTokenizer(**context):
	client = MlflowClient()
	#model_name='Alpha_{}'.format()
	run_id = context['task_instance'].xcom_pull(task_ids="getBestRun", key="bestRun")
	#model_name='Alpha_{}'.format(runid)
	#stage='Staging'
	#model = mlflow.pyfunc.load_model(
	#		model_uri='models:/{}/{}'.format(model_name, stage)
	#		)
	#run_id = model.metadata.run_id

	local_dir = './tmp'
	if not os.path.exists(local_dir):
		os.mkdir(local_dir)

	tokenizer_path = client.download_artifacts(run_id, "tokenizer.pkl", local_dir)
	tokenizer = pickle.load(open(tokenizer_path,'rb'))
	run = client.get_run(run_id)
	testSequences = tokenizer.texts_to_sequences(comments)
	paddedTestSequences = pad_sequences(testSequences,padding = run.data.params['seq_padding_style'],truncating = run.data.params['seq_truncating_style'],maxlen = int(run.data.params['max_length']))

	params = {}
	prediction = requests.post(url='http://127.0.0.1:5004/invocations',json={"inputs":paddedTestSequences.tolist()}, headers={'Content-Type':'application/json',})
	p = prediction.json()


dag = DAG(
		dag_id = "inferenceworkflow",
		#start_date = airflow.utils.dates.days_ago(3),
		start_date = pendulum.yesterday(),
		schedule_interval=None,
)


getBestRun = PythonOperator(
		task_id='getBestRun',
		python_callable = getRun,
		dag=dag
)

downloadTokenizer = PythonOperator(
		task_id = 'downloadTokenizer',
		python_callable = getTokenizer,
		dag = dag
)

loadData = PythonOperator(
		task_id = 'loadData',
		python_callable = loadAndVectorize,
		dag = dag
)
