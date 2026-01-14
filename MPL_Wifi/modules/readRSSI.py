import time
import subprocess
import pandas as pd 


def read_rssi_from_powershell(ps_script_path,duration):
    """
    PowerShell scripti çalıştırır ve stdout'tan RSSI (dBm) okur.
    Sadece TEK ölçüm içindir.
    """
    try:
        result = subprocess.run(
            ["powershell", "-ExecutionPolicy", "Bypass", "-File", ps_script_path,"-durationMinutes", str(duration)],
            capture_output=True,
            text=True,
             
            
        )

        output = result.stdout.strip()

        if output == "" or output.lower() == "nan":
            return None

        return output

    except Exception as e:
        print("RSSI read error:", e)
        return None
    



def save_excel(measurements, filename):
   
    df = pd.DataFrame(
        measurements,
        columns=["time_sec", "rssi", "room_id", "event"]
    )
    df.to_excel(filename, index=False)

def parse_rssi_string(rssi_string):
    rssi_list = []

    # satırlara böl
    lines = rssi_string.strip().split("\n")

    for line in lines:
        if not line.strip():
            continue

        # virgülle böl
        values = line.split(",")

        for v in values:
            
            rssi_list.append(float(v))

    return rssi_list



def build_measurements(rssi_list, room_id, event, sampling_period=0.25):
    measurements = []

    for i, rssi in enumerate(rssi_list):
        time_sec = round(i * sampling_period, 3)

        measurements.append([
            time_sec,
            rssi,
            room_id,
            event
        ])

    return measurements   
    
    
if __name__ == "__readRSSI__":
    read_rssi_from_powershell("../shell_script/wifi_rssi_log.ps1")