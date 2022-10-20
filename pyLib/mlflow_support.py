"""
This includes functions to support mlflow
"""

import mlflow
import shutil
from mlflow.tracking import MlflowClient


def get_experiment_id(exp_name='default', path='mlruns'):
    """Check if an experiment exists in the path, if not create it"""
    client = MlflowClient()
    experiment_id = client.get_experiment_by_name(exp_name)
    if experiment_id is None:
        experiment_id = client.create_experiment(exp_name, artifact_location=path)
    return exp_name, experiment_id


def get_run_id(experiment_id):
    """Get the run id"""
    client = MlflowClient()
    # experiment_id = get_experiment_id()[1]
    runs = client.search_runs(experiment_ids=experiment_id)
    return runs[0].info.run_id


def print_exp_info(experiment_id):
    """Prints the experiment info"""
    client = MlflowClient()
    print(f'Experiment name: {client.get_experiment(experiment_id).name}')
    print(f'Experiment id: {experiment_id}')
    print(f'Experiment runs: {len(client.list_run_infos(experiment_id))}')


def del_exp(experiment_id):
    """Deletes the experiment with mlruns folder"""
    client = MlflowClient()
    # experiment_id = get_experiment_id()[1]
    client.delete_experiment(experiment_id)
    shutil.rmtree('mlruns')


def log_mlflow_artifacts(artifact_path):
    """Logs the artifacts to mlflow"""
    mlflow.log_artifacts(artifact_path)


def log_mlflow_params(params):
    """Logs the params to mlflow"""
    mlflow.log_params(params)


def log_mlflow_metrics(metrics):
    """Logs the metrics to mlflow"""
    mlflow.log_metrics(metrics)


def log_mlflow_tags(tags):
    """Logs the tags to mlflow"""
    mlflow.set_tags(tags)


def log_mlflow_fig(fig, fig_name):
    """Logs the figure to mlflow"""
    mlflow.log_figure(fig, fig_name)


def log_mlflow_artifact(artifact_path):
    """Logs the artifact to mlflow"""
    mlflow.log_artifact(artifact_path)


def log_mlflow_run(run_name):
    """Logs the run to mlflow"""
    mlflow.start_run(run_name=run_name)


def end_mlflow_run():
    """Ends the mlflow run"""
    mlflow.end_run()


def log_mlflow_run_info(run_name, params, metrics, tags, artifact_path):
    """Logs the run info to mlflow"""
    log_mlflow_run(run_name)
    log_mlflow_params(params)
    log_mlflow_metrics(metrics)
    log_mlflow_tags(tags)
    log_mlflow_artifacts(artifact_path)
    end_mlflow_run()


