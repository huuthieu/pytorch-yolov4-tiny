import os
from random import random, randint
from mlflow import log_metric, log_param, log_artifacts
import mlflow
import argparse
import hydra

@hydra.main(config_path ="./config", config_name = "config")
def exp_track(cfg)
    # Log a parameter (key-value pair)
    
    result_file = 'result.txt'
    with open(result_file,'r') as f:
        result = f.read().splitlines()

    mAP = result[0].split('=')[0].split('%')[0]
    mAP = float(mAP)
    #rec = result[2].split(':')[1].strip('% ')
    #rec = float(rec)
    
    precision = float(result[2].split('=')[1].strip('% '))
    recall = float(result[3].split('=')[1].strip('% '))

    mlflow.set_tracking_uri(cfg.track.saved_runs)
    with mlflow.start_run(mlflow.set_experiment(cfg.track.exp_name),run_name=cfg.track.name):
        # log_param("param1", randint(0, 100))
        # Log a metric; metrics can be updated throughout the run
        # log_metric("foo", random())
        # log_metric("foo", random() + 1)
        # log_metric("foo", random() + 2)

        log_metric("mAP", mAP)
        #log_metric("Rec with Prec thresh 0.7", rec)
        log_metric("Precision", precision)
        log_metric("Recall", recall)
        # Log an artifact (output file)
        artifacts = os.path.join(cfg.train.logs, 'loss_his')
        log_artifacts(artifacts)

