from mlflow.tracking.client import MlflowClient
import mlflow
import pymongo
from pymongo import MongoClient
import json
from bson import json_util

unexecutedExperimentNames = []
with open('unexecuted.json','r') as f:
	data = json.loads(f.read())
	for d in data:
		e = json.loads(d)
		unexecutedExperimentNames.append(e['expname'])

experimentName = unexecutedExperimentNames[0]
connectionString = 'mongodb://root:password@localhost:27015/mlflowexperiments?authSource=admin'
client = MongoClient(connectionString)
db = client['mlflowexperiments']
collection = db["experimentranges"]


res_hp = collection.find({"experiment_name":experimentName})
hyperparams = ''
for r in res_hp:
	hyperparams = json_util.dumps(r)
hyperparams = json.loads(hyperparams)
popped  = hyperparams.pop("_id", None)

#print(hyperparams)
_ = mlflow.create_experiment(hyperparams['experiment_name'])

client = MlflowClient()
main_exp_id = mlflow.get_experiment_by_name('BaseExperiment').experiment_id
run = client.create_run(main_exp_id)#, tags = {"dataPath":hyperparams["dataPath"]})
mlflow.run(run_id = run.info.run_id, uri = '.', entry_point = "ht", use_conda = False, parameters = hyperparams)
