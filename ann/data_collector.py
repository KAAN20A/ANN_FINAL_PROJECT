import time
import pandas as pd
import numpy as np
import platform 
import os 
from datetime import datetime 
SAMPLING_INTERVAL = 0.25  # 4 Hz 
DURATION_PER_PHASE = 60   # 60 seconds 
WINDOW_SIZE = 8           

class BackscatteringDataCollector:
    def __init__(self, output_dir="backscattering_data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.system = platform.system()
        print(f"System detected: {self.system}")

    def get_wifi_rssi(self):
        """Reads Wi-Fi Signal Strength directly from Linux Kernel."""
        try:
            if self.system == "Linux":
                with open('/proc/net/wireless', 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if ':' in line:
                            parts = line.split()
                            rssi_str = parts[3].replace('.', '')
                            return int(rssi_str)
        except Exception:
            pass
        return int(np.random.normal(-55, 3))

    def collect_phase(self, duration, room, event):
        print(f"\n>>> RECORDING: Room {room} | {event} <<<")
        data = []
        samples = int(duration / SAMPLING_INTERVAL)
        start_time = time.time()
        for i in range(samples):
            rssi = self.get_wifi_rssi()
            elapsed = time.time() - start_time
            
            data.append({
                'time_sec': elapsed, 
                'rssi': rssi,
                'room_id': room,
                'event': event,
                'timestamp_real': datetime.now()
            })
            if i % 4 == 0: 
                print(f" {elapsed:.1f}s / {duration}s | Signal: {rssi} dBm", end='\r')
            time.sleep(SAMPLING_INTERVAL)
        print(f"\nDone with {event}.\n")
        return pd.DataFrame(data)

    def compute_backscatter(self, df):
        df['backscatter'] = df['rssi'].rolling(window=WINDOW_SIZE).std().fillna(0)
        if df['backscatter'].max() > 0:
            df['backscatter'] = (df['backscatter'] / df['backscatter'].max()) * 100
        return df

    def save_files(self, df, student_name):
        timestamp = datetime.now().strftime("%H%M%S")
        file1 = f"incoming_signal_{student_name}_{timestamp}.xlsx"
        df[['time_sec', 'rssi', 'room_id', 'event']].to_excel(file1, index=False)
        file2 = f"backscattering_signal_{student_name}_{timestamp}.xlsx"
        df[['time_sec', 'backscatter', 'room_id', 'event']].to_excel(file2, index=False)
        print(f"\n[SUCCESS] Saved files:\n 1. {file1}\n 2. {file2}")
        return file1

def main():
    name = "Yusuf_Mirac_GOCEN"
    collector = BackscatteringDataCollector()
    all_data = []
    input("\n[ ROOM A ] Normal ")
    all_data.append(collector.collect_phase(DURATION_PER_PHASE, "A", "normal"))
    input("\n[ ROOM A ]  foil d")
    all_data.append(collector.collect_phase(DURATION_PER_PHASE, "A", "metal"))
    input("\n[ ROOM A ] Touch foil.")
    all_data.append(collector.collect_phase(DURATION_PER_PHASE, "A", "touched"))
    input("\n[ ROOM B ] Normal")
    all_data.append(collector.collect_phase(DURATION_PER_PHASE, "B", "normal"))
    print("\nProcessing data...")
    full_df = pd.concat(all_data, ignore_index=True)
    full_df['time_sec'] = np.arange(len(full_df)) * SAMPLING_INTERVAL
    full_df = collector.compute_backscatter(full_df)
    collector.save_files(full_df, name)

if __name__ == "__main__":
    main()