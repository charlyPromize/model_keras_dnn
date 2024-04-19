import time
import json
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import os





def init():
   
  
   
   global ble_dnn_model 
   ble_dnn_model = keras.models.load_model('/var/azureml-app/azureml-models/ble_dnn_model_v2/1/ble_dev_model_cloud_1p_v2.keras')
   
   

def run(input_data):
   input_data = json.loads(input_data)
   input = pd.DataFrame([np.array(list(input_data.values()))])
   test_prediction = ble_dnn_model.predict(input).flatten()

   
   return  {"score":test_prediction}
