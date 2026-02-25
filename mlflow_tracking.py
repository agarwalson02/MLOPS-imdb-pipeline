import mlflow
from mlflow.tracking import MlflowClient
import dagshub
mlflow.set_tracking_uri("https://dagshub.com/agarwalson02/MLOPS-imdb-pipeline.mlflow")
dagshub.init(repo_owner='agarwalson02',repo_name='MLOPS-imdb-pipeline',mlflow=True)


client = MlflowClient()
artifacts = client.list_artifacts("1d372ca524d542978a5fea3feaf60a9d", path="model")
print("Hello")
print(artifacts)

for a in artifacts:
    print(a.path)