#base(python 3.10.9)
#tracker/ botsort.yaml 수정 후 실행


import os
import cv2
import time
import torch
import numpy as np
import pandas as pd
import ultralytics
import platform
import playsound #1.2.2

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

#------------------------------------------------------------------------------------------------------------------------------------------------------------------

CCTV_lst = []
CCTV_df = pd.DataFrame(CCTV_lst, columns = ['time', 'year', 'month', 'day', 'hour', 'person', 'accumulate'])
difference = False
id_difference = 0
ids_lst = []

model = YOLO('8s_50_32_v3.pt')
vid_path = 'CCTV_src/vid0.mp4'

#------------------------------------------------------------------------------------------------------------------------------------------------------------------

def check_id(ids, ids_lst):
    global difference
    global id_difference
    
    id_difference = 0
    id_difference = [x for x in ids if x not in ids_lst]
    if len(id_difference) >= 1:
        difference = True


def CCTV():
  global model
  global success
  global frame
  global difference
  global id_difference
  global ids_lst
  global cnt
  global CCTV_df
  results = []
  ids = []
  converted_ids = []

  time.sleep(0.1)
  results = model.track(frame, stream=True, persist=True, conf=0.2)
  for result in results:
    c_lst = result.boxes.cls
    ids = result.boxes.id
    time.sleep(0.1)
    converted_ids = ids.tolist()
    converted_ids = list(map(int, converted_ids))

  if len(c_lst) >= 1:
    curr_time = datetime.now()
    time_format = curr_time.strftime('%Y-%m-%d  %H : %M : %S')
    playsound.playsound('CCTV_utils/warning.mp3') #경보방송
    check_id(converted_ids, ids_lst)
    if difference == False: #same id
      try:  #same id
        log_df = pd.DataFrame({'time':[time_format], 'year':[curr_time.year], 'month':[curr_time.month], 'day':[curr_time.day], 'hour':[curr_time.hour],  'person':[1], 'accumulate':[CCTV_df['accumulate'].iloc[-1]]})
        CCTV_df = pd.concat([CCTV_df, log_df], ignore_index=True)
      except: #same id(frist time)
        log_df = pd.DataFrame({'time':[time_format], 'year':[curr_time.year], 'month':[curr_time.month], 'day':[curr_time.day], 'hour':[curr_time.hour], 'person':[1], 'accumulate':[0]})
        CCTV_df = pd.concat([CCTV_df, log_df], ignore_index=True)
    else: #diffrent id
      ids_lst += id_difference
      id_difference_len = len(id_difference)
      cnt += id_difference_len
      log_df = pd.DataFrame({'time':[time_format], 'year':[curr_time.year], 'month':[curr_time.month], 'day':[curr_time.day], 'hour':[curr_time.hour], 'person':[1], 'accumulate':[cnt]})
      CCTV_df = pd.concat([CCTV_df, log_df], ignore_index=True)
      id_difference_len = 0

  else:
    curr_time = datetime.now()
    time_format = curr_time.strftime('%Y-%m-%d  %H : %M : %S')
    log_df = pd.DataFrame({'time':[time_format], 'year':[curr_time.year], 'month':[curr_time.month], 'day':[curr_time.day], 'hour':[curr_time.hour], 'person':[0], 'accumulate':[CCTV_df['accumulate'].iloc[-1]]})
    CCTV_df = pd.concat([CCTV_df, log_df], ignore_index=True)


  save_time_format = curr_time.strftime('%Y-%m-%d')
  CCTV_df.to_csv(f'CCTV_DB/CCTV_result{save_time_format}.csv')
  return CCTV_df

#------------------------------------------------------------------------------------------------------------------------------------------------------------------

def crime_predictor():
  df0 = pd.read_csv('CCTV_DB/sample12.csv', index_col=0, parse_dates=True)
  df0 = df0.drop(['year','day','hour','month','temp','person'], axis=1)
  df0.reset_index(drop=False,inplace=True)

  stock_data = df0.groupby([pd.Grouper(key='date',freq='2h')]).max()
  testdata2 = df0.groupby([pd.Grouper(key='date',freq='2h')]).max()
  testdata2.reset_index(drop=False,inplace=True)
  original_open = stock_data['accum'].values

  dates = pd.to_datetime(testdata2['date'])
  cols = list(testdata2)[1:2]
  stock_data = stock_data[cols].astype(float)

  scaler = StandardScaler()
  scaler = scaler.fit(stock_data)
  stock_data_scaled = scaler.transform(stock_data)

  n_train = int(0.9*stock_data_scaled.shape[0])
  train_data_scaled = stock_data_scaled[0: n_train]
  train_dates = dates[0: n_train]

  test_data_scaled = stock_data_scaled[n_train:]
  test_dates = dates[n_train:]

  pred_days = 1 
  seq_len = 14 
  input_dim = 1  

  trainX = []
  trainY = []
  testX = []
  testY = []

  for i in range(seq_len, n_train-pred_days +1):
      trainX.append(train_data_scaled[i - seq_len:i, 0:train_data_scaled.shape[1]])
      trainY.append(train_data_scaled[i + pred_days - 1:i + pred_days, 0])

  for i in range(seq_len, len(test_data_scaled)-pred_days +1):
      testX.append(test_data_scaled[i - seq_len:i, 0:test_data_scaled.shape[1]])
      testY.append(test_data_scaled[i + pred_days - 1:i + pred_days, 0])

  trainX, trainY = np.array(trainX), np.array(trainY)
  testX, testY = np.array(testX), np.array(testY)

  model = Sequential()
  model.add(LSTM(64, input_shape=(trainX.shape[1], trainX.shape[2]),
                return_sequences=True))
  model.add(LSTM(32, return_sequences=False))
  model.add(Dense(trainY.shape[1]))

  learning_rate = 0.01
  optimizer = Adam(learning_rate=learning_rate)
  model.compile(optimizer=optimizer, loss='mse')


  prediction = model.predict(testX)
  mean_values_pred = np.repeat(scaler.mean_[np.newaxis, :], prediction.shape[0], axis=0)
  mean_values_pred[:, 0] = np.squeeze(prediction)
  y_pred = scaler.inverse_transform(mean_values_pred)[:,0]

  mean_values_testY = np.repeat(scaler.mean_[np.newaxis, :], testY.shape[0], axis=0)
  mean_values_testY[:, 0] = np.squeeze(testY)
  testY_original = scaler.inverse_transform(mean_values_testY)[:,0]

  try:
    model.load_weights('lstm_weights.h5')
  except:
      print("No weights found, training model from scratch")
      history = model.fit(trainX, trainY, epochs=50, batch_size=32, validation_split=0.1, verbose=1)
      model.save_weights('lstm_weights.h5')

  zoom_start = len(test_dates) - 50
  zoom_end = len(test_dates)

  plt.figure(figsize=(14, 5))
  adjusted_start = zoom_start - seq_len

  plt.plot(test_dates[zoom_start:zoom_end],
          testY_original[adjusted_start:zoom_end - zoom_start + adjusted_start],
          color='blue',
          label='Actual People')

  plt.plot(test_dates[zoom_start:zoom_end],
          y_pred[adjusted_start:zoom_end - zoom_start + adjusted_start ],
          color='red',
          linestyle='--',
          label='Predicted People')

  plt.xlabel('Date')
  plt.ylabel('Open People')
  plt.title('Zoomed In Actual vs Predicted People')
  plt.legend()
  plt.savefig('/static/img/lstm.png')    

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------


cap = cv2.VideoCapture(vid_path)
cnt = 0

#약 18초마다 2초씩 객체탐지 수행
fps = 60
interval = fps * 20 #20-2 = 18
frame_counter = 0

while cap.isOpened():
  success, frame = cap.read()
  if not success:
    break
  frame_counter += 1
  if frame_counter % interval == 0:
    start_time = time.time()
    while (time.time() - start_time) <= 2:
      CCTV()

cap.release()
cv2.destroyAllWindows()


