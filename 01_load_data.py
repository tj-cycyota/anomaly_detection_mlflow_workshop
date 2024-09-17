# Databricks notebook source
# MAGIC %md
# MAGIC #### Setup
# MAGIC This code has been created and tested for Databricks Runtime `15.4.x-cpu-ml-scala2.12`. Please make sure your cluster conforms to this version. 

# COMMAND ----------

# MAGIC %run ./_config

# COMMAND ----------

display(dbutils.fs.ls("dbfs:/databricks-datasets/iot-stream/data-device/"))

# COMMAND ----------

# Copy Files
try:
  print(f"Copying source data to: {user_folder}{raw}")
  dbutils.fs.cp(source_folder, user_folder+raw, recurse=True)
except:
  print(f"Deleting old files, and copying source data to: {user_folder}")
  dbutils.fs.rm(user_folder, recurse=True)
  dbutils.fs.cp(source_folder, user_folder+raw, recurse=True)

# Create dirs
try:
  print(f"Creating dirs: {user_folder}{raw_clean}, {user_folder}{landing}")
  dbutils.fs.mkdirs(user_folder+raw_clean)
  dbutils.fs.mkdirs(user_folder+landing)
except:
  print(f"Resetting directories: {user_folder}{raw_clean}, {user_folder}{landing}")
  dbutils.fs.rm(user_folder+raw_clean, True)
  dbutils.fs.rm(user_folder+landing, True)
  dbutils.fs.mkdirs(user_folder+raw_clean)
  dbutils.fs.mkdirs(user_folder+landing)
  
display(dbutils.fs.ls(user_folder+raw))

# COMMAND ----------

# Inject anomalies into every file
import pandas as pd
import numpy as np
for file in dbutils.fs.ls(user_folder+raw):
  filepath = file.path
  
  print(f"Modifying {filepath} to inject anomalies")
  data = spark.read.json(filepath).toPandas()
  data["anomaly"] = 1 # Isolation forests use 1 for inliers
  
  # Sample .005% of rows
  rows = len(data)
  random_rows = np.random.choice(data.index, size=np.floor(rows*0.005).astype(int), replace=False)

  # Replace with random multiple of current value
  data.loc[random_rows, "calories_burnt"] = data.loc[random_rows, "calories_burnt"] * 2
  data.loc[random_rows, "miles_walked"] = data.loc[random_rows, "miles_walked"] * 0.5
  data.loc[random_rows, "anomaly"] = -1 # -1 for outliers

  # Rename and downselect only relevant columns
  data_final = data.rename(columns={
    "calories_burnt": "signal_1",
    "miles_walked": "signal_2",
    "num_steps": "signal_3"
  })[['timestamp','device_id','signal_1','signal_2','signal_3','anomaly']]

  # Save data back to exact same filepath in cleaned dir
  filepath_clean = filepath.replace('dbfs:/','/dbfs/').replace('raw','clean')
  print(f"Writing file to: {filepath_clean}")
  data_final.to_json(filepath_clean, orient='records', lines=True)
  print("Successfully wrote file.")
  print(".......................")


# COMMAND ----------

# Confirm we can read all files
from pyspark.sql.functions import to_timestamp

full_data = (spark.read.json(user_folder+raw_clean)
             .withColumn('timestamp', to_timestamp("timestamp", 'yyyy-MM-dd HH:mm:ss.SSSSSS'))
)

display(full_data)

display(full_data.count())


# COMMAND ----------

display(dbutils.fs.ls(user_folder+raw_clean))
