import cv2, numpy as np
import pylab as pl
import pandas as pd
from pykalman import KalmanFilter
import matplotlib.pyplot as plt
import csv
import time


meas=[]
pred=[]
frame = np.zeros((800,800,3), np.uint8) # drawing canvas
mp = np.array((5,1), np.float32) # measurement
tp = np.zeros((5,1), np.float32) # tracked / prediction

# input data Filter
kalman = cv2.KalmanFilter(5,5)
kalman.measurementMatrix = np.array([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]],np.float32)
kalman.processNoiseCov = np.array([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]],np.float32) * 0.03

#pred.append((int(tp[0]),int(tp[1])))
with open('dataset.csv', 'rb') as csvfile:
     spamreader = csv.reader(csvfile, delimiter=';', quotechar='|')
     i=0
     for row in spamreader:
        #print 'row1  ', row[1], "row3: ", row[3]
        mp = np.array([np.float32(float(row[1])),np.float32(float(row[2])),np.float32(float(row[3])),np.float32(float(row[4])),np.float32(float(row[5]))])
        kalman.correct(mp)
        tp = kalman.predict()
        pred.append((tp[0],tp[1],tp[2],tp[3],tp[4]))
        i=i+1



data = np.asarray(pred)
#print data[:,1,0]
input_file = "dataset.csv"
dataset = pd.read_csv(input_file, header = 0, delimiter = ";")
dataset = dataset.as_matrix()
#print dataset[:,0]
    #print pred[][0]
plt.plot(dataset[:,0], data[:-1,1,0],'r-',dataset[:,0], dataset[:,2],'b-')
plt.show()
