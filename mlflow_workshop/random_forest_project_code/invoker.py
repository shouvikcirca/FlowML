import mlflow
from mlflow.tracking.client import MlflowClient

# Create Experiment
experiment_id = mlflow.create_experiment("RFExperiment")

client = MlflowClient()

# Create run
run = client.create_run(experiment_id)

# Within the created run, invoke rf entrypoint
mlflow.run(run_id = run.info.run_id, uri = '.', entry_point = "rf", use_conda = False)
