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

all_raw_data = []
all_backscatter_data = []
WINDOW_SIZE = 8
WINDOW = 10
class ANNModel:
    def __init__(self):
        self.OS=platform.system()
        self.model_path=None
    

    def plot_signal(df, signal_column, time_column='time_sec', title=None):
    
        plt.figure(figsize=(10, 5))
        plt.plot(df[time_column], df[signal_column], label=signal_column, color='blue')
        plt.xlabel('Time (s)')
        plt.ylabel('Signal')
        plt.title(title if title else f"{signal_column} over time")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    def plot_room_event_matrix(df, room_col="room_id", event_col="event"):
   
    
        matrix = df.pivot_table(index=event_col, columns=room_col, aggfunc='size', fill_value=0)

        plt.figure(figsize=(8, 5))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
        plt.ylabel("Event")
        plt.xlabel("Room")
        plt.title("Room vs Event Measurement Counts")
        plt.tight_layout()
        plt.show()
    
        return matrix



        

    def load(self):
        """Kaydedilmiş modeli yükle"""
        model_file = "../current_model/ann_backscatter_model.keras"
        if model_file and os.path.exists(model_file):
            self.model = tf.keras.models.load_model(model_file)
            print(f"Model yüklendi: {model_file}")
        else:
            raise FileNotFoundError(f"Model dosyası bulunamadı: {model_file}")

    def predict(self, data):
        """
        Veri üzerinde tahmin yapar.
        data: tek bir örnek (1D array) veya birden fazla örnek (2D array)
        """
        room_labels = {0: "Room A", 1: "Room B"}
        event_labels = {0: "Normal", 1: "Metal", 2: "Touched"}
        if self.model is None:
            raise ValueError("Model yüklenmedi. Önce load() veya build() çağırın.")

        

        
        
        data2 =[
            'backscatter',
            'bs_mean',
            'bs_std',
            'bs_diff',
            'bs_energy'
        ]
        data2 = pd.DataFrame({'backscatter': data})

        data2['backscatter'] = data2['backscatter']
        data2['bs_mean'] = data2['backscatter'].rolling(WINDOW).mean().fillna(0)
        data2['bs_std']  = data2['backscatter'].rolling(WINDOW).std().fillna(0)
        data2['bs_diff'] = data2['backscatter'].diff().fillna(0)
        data2['bs_energy'] = data2['backscatter'].rolling(WINDOW).apply(lambda x: np.sum(x**2)).fillna(0)

       
        predictions = self.model.predict(data2)

        
        if isinstance(predictions, list) and len(predictions) == 2:
            room_pred, event_pred = predictions
            room_class = np.argmax(room_pred, axis=1)
            event_class = np.argmax(event_pred, axis=1)
            room_class_str  = [room_labels[i] for i in room_class]
            event_class_str = [event_labels[i] for i in event_class]
            most_common_room = Counter(room_class_str).most_common(1)[0][0]
            most_common_event = Counter(event_class_str).most_common(1)[0][0]
            return most_common_room, most_common_event
        else:
            
            return np.argmax(predictions, axis=1)

    def extract_features(self,room_id, event,duration):
        """
        Incoming signal + backscattering datasından
        ANN için feature vector üretir

        Beklenen kolonlar:
        - time_sec
        - rssi
        - backscatter
        """
        WINDOW_SIZE=8
        
        if self.OS=="Windows":
            incoming_signal=rR.read_rssi_from_powershell("../shell_scripts/wifi_rssi_log.ps1",duration)
            incoming_signal=rR.parse_rssi_string(incoming_signal)
            measurements=rR.build_measurements(incoming_signal,room_id,event, sampling_period=0.25)
          
        elif self.OS=="Linux":
            incoming_signal=rR.read_rssi_from_powershell("../shell_scripts/wifi_rssi_log_gnu.sh",duration)
            incoming_signal=rR.parse_rssi_string(incoming_signal)
            measurements=rR.build_measurements(incoming_signal,room_id,event, sampling_period=0.25)
         
        else: #macos(darwin)
            incoming_signal=rR.read_rssi_from_powershell("../shell_scripts/wifi_rssi_log_macos.sh",duration)
            incoming_signal=rRparse_rssi_string(incoming_signal)
            measurements=rR.build_measurements(incoming_signal,room_id,event, sampling_period=0.25)
          
        

        df = pd.DataFrame(
        measurements,
        columns=["time_sec", "rssi", "room_id", "event"]
        )
        df['rssi'] = df['rssi'].rolling(window=WINDOW_SIZE).std().fillna(0)
       

       

        return measurements,df
    def run_phase(self,room_id, event, duration, instructions,time):
        print("\n" + "-"*60)
        print(f"ROOM {room_id} | EVENT: {event.upper()}")
        print("-"*60)
        print(instructions)
        input("Press ENTER to start measurement...")

            # Ham ölçüm
           # df=rR.read_rssi_from_powershell("../shell_scripts/wifi_rssi_log.ps1")
            #df = rR.build_measurements(df, room_id, event, sampling_period=0.25)
            #df=rR.parse_rssi_string(df)
            
            # Feature extraction
            #backscatter_df = rR.build_measurements(df, room_id, event, sampling_period=0.25)
        df,backscatter_df=self.extract_features(room_id,event,time)
        return df,backscatter_df

    def run_ann_experiment(self):
        print("\n" + "="*70)
        print("ANN DATA COLLECTION PROTOCOL")
        print("="*70)
        print("Rooms: A, B, C")
        print("States:")
        print(" - Room A/B: normal, metal, touch")
        print(" - Room C: normal only")
        print("="*70)

        

        
    # =======================
    # ROOM A
    # =======================
        df_temp,backscatter_df_temp=self.run_phase(
            room_id="A",
            event="normal",
            duration=60,
            instructions="Normal usage. No metal nearby.",
            time=1
        )
        df=df_temp
        backscatter_df=backscatter_df_temp
        df_temp,backscatter_df_temp=self.run_phase(
            room_id="A",
            event="metal",
            duration=60,
            instructions="Place aluminum foil ~30cm from laptop.",
            time=1
        )
        df=df+df_temp
        backscatter_df=backscatter_df+backscatter_df_temp

        df_temp,backscatter_df_temp=self.run_phase(
            room_id="A",
            event="touch",
            duration=60,
            instructions="Touch or slightly move the foil occasionally.",
            time=1
        )
        df=df+df_temp
        backscatter_df=backscatter_df+backscatter_df_temp
        #raw_dataset = pd.concat(all_raw_data, ignore_index=True)
        #backscatter_dataset = pd.concat(all_backscatter_data, ignore_index=True)

        
        
       # all_raw_data = pd.concat(all_raw_data, ignore_index=True)
        #all_backscatter_data = pd.concat(all_backscatter_data, ignore_index=True)
        
    # =======================
    # ROOM B
    # =======================
    
        df_temp,backscatter_df_temp=self.run_phase(
            room_id="B",
            event="normal",
            duration=60,
            instructions="Normal usage in Room B.",
            time=1
        )
        df=df+df_temp
        backscatter_df=backscatter_df+backscatter_df_temp
        df_temp,backscatter_df_temp=self.run_phase(
            room_id="B",
            event="metal",
            duration=60,
            instructions="Place aluminum foil ~30cm from laptop.",
            time=1
        )
        df=df+df_temp
        backscatter_df=backscatter_df+backscatter_df_temp
        df_temp,backscatter_df_temp=self.run_phase(
            room_id="B",
            event="touch",
            duration=60,
            instructions="Touch or slightly move the foil occasionally.",
            time=1
        )
        df=df+df_temp
        backscatter_df=backscatter_df+backscatter_df_temp
    # =======================
    # ROOM C (baseline only)
    # =======================
        df,backscatter_df=self.run_phase(
            room_id="C",
            event="normal",
            duration=60,
            instructions="Baseline measurement in Room C. No metal.",
            time=1
        )
        df=df+df_temp
        backscatter_df=backscatter_df+backscatter_df_temp
    # =======================
    # CONCATENATE DATA
    # =======================
        #raw_dataset = pd.concat(all_raw_data, ignore_index=True)
        #backscatter_dataset = pd.concat(all_backscatter_data, ignore_index=True)

        print("\n" + "="*70)
        print("DATA COLLECTION COMPLETE")
        print("="*70)
        #print(f"Total raw samples: {len(raw_dataset)}")
        #print(f"Total backscatter samples: {len(backscatter_dataset)}")
        print("="*70)
        rR.save_excel(df, "kaan_acar_incoming_signal.xlsx")
        rR.save_excel(backscatter_df, "kaan_acar_backscattering_signal.xlsx")
        return df, backscatter_df
    

   


if __name__ == "__ANNModel__":
    
  
   
   
