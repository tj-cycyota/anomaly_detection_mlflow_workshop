# Databricks notebook source
# MAGIC %md
# MAGIC # 02 - Train Isolation Forest Model
# MAGIC There are many approaches to anomaly detection, but we'll use a relatively basic one first. We will focus on properly tracking the experiment with MLflow so we can use it for inference at a later time (and in later notebooks). We'll take a `crawl -> walk -> run` approach, moving from simpler techniques and concepts to more complex approaches.
# MAGIC
# MAGIC > The IsolationForest ‘isolates’ observations by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature.
# MAGIC
# MAGIC ![iso forest](https://scikit-learn.org/stable/_images/sphx_glr_plot_isolation_forest_002.png)
# MAGIC
# MAGIC #### Setup
# MAGIC This code has been created and tested for Databricks Runtime `15.4.x-cpu-ml-scala2.12`. Please make sure your cluster conforms to this version. 
# MAGIC
# MAGIC Inspiration: https://www.databricks.com/blog/near-real-time-anomaly-detection-delta-live-tables-and-databricks-machine-learning
# MAGIC
# MAGIC Resource: https://victoriametrics.com/blog/victoriametrics-anomaly-detection-handbook-chapter-3/

# COMMAND ----------

# MAGIC %run ./_config

# COMMAND ----------

import pandas as pd
from pyspark.sql.functions import to_timestamp, unix_timestamp
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import mlflow
from mlflow.models.signature import infer_signature
mlflow.autolog(disable=True) #Disable autologging to show manual

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load data prepared in first notebook

# COMMAND ----------

full_data = (spark.read
              .json(user_folder+raw_clean+"part-00000.json.gz")
              .withColumn('timestamp', to_timestamp("timestamp", 'yyyy-MM-dd HH:mm:ss.SSSSSS'))
              .withColumn('epoch', unix_timestamp('timestamp'))
              .drop('timestamp').drop('device_id')
              .toPandas()
              )

# COMMAND ----------

display(full_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Train/Test Split

# COMMAND ----------

train, test = train_test_split(full_data, random_state=123)
X_train = train.drop(columns="anomaly")
X_test = test.drop(columns="anomaly")
y_train = train["anomaly"]
y_test = test["anomaly"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Fit & Evaluate Isolation Forest Model (crawl)

# COMMAND ----------


isolation_forest = IsolationForest(
    n_estimators=1000, 
    contamination=0.005, 
    n_jobs=-1, 
    warm_start=True, 
    random_state=42)
    
isolation_forest.fit(X_train)

# COMMAND ----------

y_pred_train = isolation_forest.predict(X_train)
print(classification_report(y_train, y_pred_train))

# COMMAND ----------

# MAGIC %md
# MAGIC Join predictions back to true labels to "eyeball" which ones we correctly/incorrectly labeled as anomalies. 
# MAGIC
# MAGIC Remember, `-1` are outliers/anomalies, `1` are inliers/normal

# COMMAND ----------

train_reset = train.reset_index(drop=True)
y_pred_train_df = pd.DataFrame(y_pred_train, columns=['y_pred'])
display(pd.concat([train_reset, y_pred_train_df], axis=1))

# COMMAND ----------

# MAGIC %md
# MAGIC Score on held-out test data

# COMMAND ----------

y_pred_test = isolation_forest.predict(X_test)
print(classification_report(y_test, y_pred_test))

# COMMAND ----------

# MAGIC %md
# MAGIC Lots to do to improve this model! Normal data science recommendations apply here: perform feature engineering, hyperparameter optimization, more data to train with, etc. We'll skip that for now, as the anomalies in this dataset are totally made up. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Log Model to MLflow (walk)
# MAGIC We will train our model again, but this time log it to MLflow. MLflow is critical as it allows us to track everything about our data science process. 
# MAGIC
# MAGIC In the cell below, notice the `mlflow...log_model()` command. This is what is physically saving our model to a storage location (the MLflow Tracking Server) for later use. You can use that command to register a model object to a run and then decide to register it later; but in this case, we are going to register this as a model as we log it. 
# MAGIC
# MAGIC We turned off [MLflow Autologging](https://docs.databricks.com/en/mlflow/databricks-autologging.html#disable-databricks-autologging) so that we can show how to manually control what about our experiment is logged. Later on we can re-enable autologging, which is the default in Databricks.
# MAGIC
# MAGIC If you see MLflow warnings below, you can safely ignore them as long as you see a run logged to your MLflow experiment.

# COMMAND ----------

# Always start an mlflow run with context management, e.g. "with" clause
with mlflow.start_run(run_name=f"{run_counter}_basic_{run_name}") as run:
    # Capture useful information for MLflow to log
    signature = infer_signature(X_train, y_pred_train)
    
    mlflow.sklearn.log_model(
        isolation_forest, 
        artifact_path="model",
        signature=signature,
        registered_model_name = registered_model_name
    )

    # Can also log hyperparams and metrics of the training
    run_id = run.info.run_id
    print("run_id",run_id)
    # Increment run counter
    run_counter+=1

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load IsoForest model back and evaluate on test data
# MAGIC At this point, we have permanently saved our model to an MLflow Experiment. If our cluster terminates or we want to access the model from a different compute source, we can easily do that with MLflow. Let's demonstrate by evaluating on our test set.
# MAGIC

# COMMAND ----------

loaded_model = mlflow.sklearn.load_model(model_uri = f"runs:/{run_id}/model")

y_pred_test_2 = loaded_model.predict(X_test)
print(classification_report(y_test, y_pred_test_2))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Change registered model status to "Staging"
# MAGIC In the Model Registry UI (left-nav, click `Model`), change the status of your model to `Staging` stage. The cell below demonstrates how to load a model back for inference from a particular stage (rather than with just a run ID)
# MAGIC
# MAGIC Take a moment to review the various model stages (e.g. `Production`, `Staging`, `Archived`, `None`):

# COMMAND ----------

# client = mlflow.tracking.MlflowClient()
# # Retrieve version of the model we just logged
# model_version = mlFlowClient.get_latest_versions(model_name,stages=['None'])[0].version
# # Transition this model to production (you can also automate testing by transitioning to "Staging")
# mlFlowClient.transition_model_version_stage(
#     name= model_name, 
#     version = model_version, 
#     stage='Staging', 
#     archive_existing_versions= True
# )
    
#     # Return resource URI
# resource_uri = mlFlowClient.get_latest_versions(model_name, stages=["Production"])[0].source

# COMMAND ----------

staging_model = mlflow.sklearn.load_model(model_uri = f"models:/{registered_model_name}/staging")

y_pred_test_3 = staging_model.predict(X_test)
print(classification_report(y_test, y_pred_test_3))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Advanced Model development (run)
# MAGIC
# MAGIC This is with a different synthetic dataset, but demonstrates how to use Pytorch 
# MAGIC
# MAGIC * Re-enable MLflow autologging:
# MAGIC > Unlike other deep learning flavors, MLflow does not have an autologging integration with PyTorch because native PyTorch requires writing custom training loops.
# MAGIC * Use a more sophisticated Anomaly Detection model
# MAGIC
# MAGIC References: 
# MAGIC * (MLflow for Torch) https://mlflow.org/docs/latest/deep-learning/pytorch/guide/index.html
# MAGIC * (Keras) https://github.com/yinxi-db/dais-2022-AE-demo/tree/master
# MAGIC * (Pytorch) https://www.geeksforgeeks.org/how-to-use-pytorch-for-anomaly-detection/
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %pip install torchinfo

# COMMAND ----------

import numpy as np
np.random.seed(0) # Seed for reproducibility
import matplotlib.pyplot as plt
import torch
from torch import nn
from torchinfo import summary
import mlflow
mlflow.pytorch.autolog()
mlflow.autolog(disable=False)

# COMMAND ----------

# Generate synthetic time-series data
data_length = 300
data = np.sin(np.linspace(0, 20, data_length)) + np.random.normal(scale=0.5, size=data_length)
# Introduce anomalies
data[50] += 6  # Anomaly 1
data[150] += 7 # Anomaly 2
data[250] += 8 # Anomaly 3

# COMMAND ----------

# Function to create sequences
def create_sequences(data, window_size):
    sequences = []
    for i in range(len(data) - window_size):
        sequences.append(data[i:i+window_size])
    return np.array(sequences)

window_size = 10
sequences = create_sequences(data, window_size)
print(sequences)

# COMMAND ----------

# Define the Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(window_size, 5),
            nn.ReLU(),
            nn.Linear(5, 2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 5),
            nn.ReLU(),
            nn.Linear(5, window_size),
            nn.ReLU()
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = Autoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Convert sequences to PyTorch tensors
sequences = torch.tensor(sequences, dtype=torch.float32)

# COMMAND ----------

# Training loop
num_epochs = 100

with mlflow.start_run(run_name=f"{run_counter}_pytorch_{current_user_safe}") as run:
    params = {
        "epochs": num_epochs,
        "learning_rate": 1e-3,
        "optimizer": "Adam"
    }
    mlflow.log_params(params) # Log training parameters.

    # Log model architecture summary.
    with open("model_summary.txt", "w") as f:
        f.write(str(summary(model)))
    mlflow.log_artifact("model_summary.txt")

    # Training loop
    model.train() # Set model to training mode
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(sequences)
        loss = criterion(output, sequences)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')
            mlflow.log_metric("loss", f"{loss.item():3f}", step=(epoch // 10))

    # Evaluation 
    with torch.no_grad():
        predictions = model(sequences)
        losses = torch.mean((predictions - sequences)**2, dim=1)
    
    metrics = {
        "losses_mean":  losses.mean(),
        "losses_std": losses.std(),
        "threshold": losses.mean() + 2 * losses.std()
    }
    mlflow.log_metrics(metrics)

    # Infer Signature
    signature = mlflow.models.infer_signature(
        sequences.numpy(),
        model(sequences).detach().numpy(),
    )

    # Log model
    mlflow.pytorch.log_model(model, "model", signature=signature)

# COMMAND ----------

# Anomaly detection
with torch.no_grad():
    predictions = model(sequences)
    losses = torch.mean((predictions - sequences)**2, dim=1)
    plt.hist(losses.numpy(), bins=50)
    plt.xlabel("Loss")
    plt.ylabel("Frequency")
    plt.show()

# Threshold for defining an anomaly
threshold = losses.mean() + 2 * losses.std()
print(f"Anomaly threshold: {threshold.item()}")

# Detecting anomalies
anomalies = losses > threshold
anomaly_positions = np.where(anomalies.numpy())[0]
print(f"Anomalies found at positions: {np.where(anomalies.numpy())[0]}")

# Plotting anomalies on the time-series graph
plt.figure(figsize=(10, 6))
plt.plot(data, label='Data')
plt.scatter(anomaly_positions, data[anomaly_positions], color='r', label='Anomaly')
plt.title("Time Series Data with Detected Anomalies")
plt.xlabel("Time Steps")
plt.ylabel("Value")
plt.legend()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load Pytorch model back for inference on new data

# COMMAND ----------

loaded_pytorch_model = mlflow.pytorch.load_model(model_uri=f"runs:/{run.info.run_id}/model")

# COMMAND ----------

new_sequence = sequences[-1]
predictions = loaded_pytorch_model(new_sequence)
print(predictions)
