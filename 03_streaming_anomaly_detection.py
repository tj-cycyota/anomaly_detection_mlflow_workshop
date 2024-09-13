# Databricks notebook source
# MAGIC %md
# MAGIC # 03 - Perform Streaming Anomaly Detection
# MAGIC
# MAGIC Our use-case will focus on detecting anomalies at low latency. That is, for our business purpose, it is critical to detect anomalies within seconds of a new record or batch of records arriving. 
# MAGIC
# MAGIC The demo below demonstrates how to perform anomaly detection as JSON files arrive in cloud storage using Spark Structured Streaming. The same approach will work for various streaming sources:
# MAGIC * Files arriving in Cloud Storage (this demo)
# MAGIC * Kafka/EventHub streaming source
# MAGIC * Delta Table streaming source

# COMMAND ----------

# MAGIC %run ./_config

# COMMAND ----------

from pyspark.sql.functions import to_timestamp, unix_timestamp, col

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Test DataFactory: 
# MAGIC This will move a file into a cloud location that we will "monitor" to simulate new files arriving.

# COMMAND ----------

datafactory.load()
# Uncomment below to reset
# datafactory.reset_landing()

# COMMAND ----------

display(dbutils.fs.ls(user_folder+landing))

# COMMAND ----------

# MAGIC %md
# MAGIC Notice below: `spark.read` for batch reads

# COMMAND ----------

new_batch_data = (spark.read
              .json(user_folder+landing+"part-00000.json.gz")
              .withColumn('timestamp', to_timestamp("timestamp", 'yyyy-MM-dd HH:mm:ss.SSSSSS'))
              .withColumn('epoch', unix_timestamp('timestamp'))
              .drop('timestamp')
              .drop('device_id')
              .drop('anomaly') # Drop label column
              )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load model for batch inference
# MAGIC
# MAGIC We'll start with simple batch inference, demonstrating a job that might run every few minutes or hours, to get a sense of how to apply a model to new data.
# MAGIC
# MAGIC Make sure you have transitioned your model to `Production` stage:

# COMMAND ----------

import mlflow
loaded_model = mlflow.pyfunc.spark_udf(
    spark, 
    model_uri=f"models:/{registered_model_name}/Production", 
    result_type='double'
)

# COMMAND ----------

# Predict on a Spark DataFrame.
columns = list(new_batch_data.columns)
new_batch_data_anomalies = new_batch_data.withColumn('prediction', loaded_model(*columns))

display(new_batch_data_anomalies)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Create a quick visualization
# MAGIC * Click `+` next to table results
# MAGIC * Chart type: `Scatter`
# MAGIC * X column -> `epoch`
# MAGIC * Y columns -> `signal_1` (can try others as well)
# MAGIC * Group by -> `prediction`
# MAGIC
# MAGIC You should see the model does OK: many anomalies are labeled, some are not, and some inliers are labeled as anomalies.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Setup AutoLoader Stream
# MAGIC
# MAGIC AutoLoader is a Databricks feature that continually loads data from a cloud storage location. It is a powerful feature for anomaly detection, as new records that arrive will be continually processed at low latency.
# MAGIC
# MAGIC [Autoloader Docs](https://learn.microsoft.com/en-us/azure/databricks/ingestion/cloud-object-storage/auto-loader/)

# COMMAND ----------

autoloader_checkpoints = user_folder+autoloader_path
print(f"AutoLoader schema tracked in:   {autoloader_checkpoints}")

landing_directory = user_folder+landing
print(f"New files arriving to:          {landing_directory}")

# COMMAND ----------

# MAGIC %md
# MAGIC When you run the cell below, it will continue to run until you terminate it. 
# MAGIC
# MAGIC **IMPORTANT** After you are done with the lab, cancel this cell or your cluster will run indefinitely.

# COMMAND ----------

streaming_df = (spark.readStream
                .format("cloudFiles")
                .option("cloudFiles.format", "json")
                .option("cloudFiles.schemaEvolutionMode", "rescue") # Only do this for demo purposes
                .option("cloudFiles.schemaLocation", autoloader_checkpoints)
                .load(landing_directory)
                .withColumn('timestamp', to_timestamp("timestamp", 'yyyy-MM-dd HH:mm:ss.SSSSSS'))
                .withColumn('epoch', unix_timestamp('timestamp'))
                .withColumn("signal_3", col("signal_3").cast("integer"))
                # .drop('timestamp')
                # .drop('device_id')
                # .drop('anomaly') 
                # .drop('_rescued_data')
                .withColumn('predictions', loaded_model(*columns))
                # .where("anomaly=True")
                # .writeStream
                # .table("target_table_name")
                )

display(streaming_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's setup a quick visualization to monitor anomalies over time. Notice how we can run normal SQL on top of our streaming dataframe. The UI can only display 1K records at a time, so its helpful to run aggregate queries on the streaming dataframe.

# COMMAND ----------

streaming_df.createOrReplaceTempView('predicted_anomalies')

display(
    spark.sql("""
        SELECT 
            COUNT(1) AS total_records, 
            SUM(CASE WHEN predictions = -1 THEN 1 ELSE 0 END) as count_anomalies
        FROM predicted_anomalies
""")
)

# COMMAND ----------

display(
    spark.sql("SELECT predictions, COUNT(*) AS count_records FROM predicted_anomalies GROUP BY ALL")
)

# COMMAND ----------

# MAGIC %md
# MAGIC Now that we have are automated stream running, we can start to load new files. 
# MAGIC
# MAGIC When you run the cell below, you should (nearly) immediately start to see new predictions about anomalies being made:

# COMMAND ----------

datafactory.load()

# COMMAND ----------

# MAGIC %md
# MAGIC Now we can turn our DataFactory on to `continuous` mode so a new file arrives every 60 seconds. Monitor the charts above to see data flowing in. 

# COMMAND ----------

datafactory.load(continuous=True)

# COMMAND ----------

display(dbutils.fs.ls(landing_directory))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write Stream to a Delta table

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## Stream from our first Delta table
# MAGIC
# MAGIC Delta tables are also a Streaming Source!

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clean Up
# MAGIC
# MAGIC Make sure you terminate the stream above or it will run indefinitely!
# MAGIC
# MAGIC Run the cell below to stop streams and cleanup checkpoint locations

# COMMAND ----------

# Get a list of all active streaming queries
queries = spark.streams.active

# Terminate all streaming queries
for query in queries:
    print("Stopping query ", query)
    query.stop()

# COMMAND ----------

# dbutils.fs.rm(landing_directory, True)
# Also delete the raw_clean, raw, and autoloader_path directories
