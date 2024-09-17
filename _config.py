# Databricks notebook source
# Setup user-specific copy of source data
current_user = spark.sql("SELECT current_user() as username").collect()[0].username
current_user_safe = current_user.split("@")[0].replace(".","_")

source_folder = "dbfs:/databricks-datasets/iot-stream/data-device/"
user_folder = f"dbfs:/Users/{current_user}/"
raw = "iot_data_raw/"
raw_clean = "iot_data_clean/"
landing = "iot_data_landing/"
autoloader_path = "checkpoints/"
stream_chkpt = "stream_checkpoints/"
stream_table = "stream_delta/"

print("user_folder:", user_folder)

# COMMAND ----------

registered_model_name = f'iforest_{current_user_safe}'
run_name = f'run_iforest_{current_user_safe}'
run_counter = 0

print(f"Model registry name: {registered_model_name}")

# COMMAND ----------

class DataFactory:
    def __init__(self, ):
        self.source = user_folder+raw_clean
        self.landing = user_folder+landing
        self.count = 0
        self.sleep_interval = 60
    
    def load(self, continuous=False):
        import time
        if self.count >= 20:
            print("Data source exhausted\n")
            
        elif continuous == True:
            while self.count < 20:
                curr_file = f"part-000{self.count:02}.json.gz"
                print(f"Loading the file {curr_file} to the landing location")
                dbutils.fs.cp(self.source + curr_file, self.landing + curr_file)
                self.count += 1
                time.sleep(self.sleep_interval) 

        else:
            curr_file = f"part-000{self.count:02}.json.gz"
            target_dir = f"{self.landing}{curr_file}"
            print(f"Loading the file {curr_file} to the landing location")
            dbutils.fs.cp(f"{self.source}{curr_file}", target_dir)
            self.count += 1

    def reset_landing(self):
        print(f"Deleting all files from landing location {self.landing}.")
        self.count = 0
        dbutils.fs.rm(self.landing, recurse=True)

datafactory = DataFactory()
print("Created datafactory for loading data")

# COMMAND ----------



# COMMAND ----------


