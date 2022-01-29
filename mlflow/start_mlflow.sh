#!/bin/bash
mlflow server --backend-store-uri postgresql://postgres:postgres@127.0.0.1:5432/postgres --default-artifact-root $HOME/mlflow_artifacts --host 0.0.0.0:5000