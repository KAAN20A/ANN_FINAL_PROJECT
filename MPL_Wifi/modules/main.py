"""
BACKSCATTERING FINAL EXAM - DATA COLLECTION SCRIPT
University: [ISTANBUL TECNICIAL UNIVERSITY]
Course: Artificial Neural Networks
Student: [Kaan Acar]
Student ID: [040230751]
Purpose: Collect WiFi RSSI data for backscattering experiments
Sampling Rate: 250ms (4 Hz) - Scientifically valid for:
 - Human motion detection (35cm movement in 250ms at walking speed)
 - Multipath variation capture
 - Comparable to published WiFi sensing systems
"""


import os
import ANNModel
import os
import tensorflow as tf  
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import platform
import readRSSI as rR
import pandas as pd
import sys
from collections import Counter
import matplotlib.pyplot as plt

def main(model_path="../current_model/latest_model.h5"):
    

    

    new_annModel=ANNModel()
    new_annModel.load()
    x = rR.read_rssi_from_powershell("../shell_scripts/wifi_rssi_log.ps1",2)
    x=rR.parse_rssi_string(x)
    x = pd.Series(x, name='rssi')
    x = pd.to_numeric(x, errors='coerce')
    room_class, event_class = new_annModel.predict(x)
    print("Tahmin edilen oda:", room_class)
    print("Tahmin edilen event:", event_class)
    

    
    


if __name__ == "__main__":
    main()
