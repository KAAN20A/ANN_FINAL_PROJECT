import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

FILENAME = "incoming_signal_Yusuf_Mirac_GOCEN_220448.xlsx"  
STUDENT_NAME = "Yusuf Mirac GOCEN"
OUTPUT_IMAGE = "academic_plots.png"

def generate_plots():
    if not os.path.exists(FILENAME):
        print(f"Error: Could not find {FILENAME}")
        return

    print(f"Loading data from {FILENAME}...")
    df = pd.read_excel(FILENAME)
    window_size = 20 
    df['backscatter'] = df['rssi'].rolling(window=window_size).std().fillna(0)
    if df['backscatter'].max() > 0:
        df['backscatter'] = (df['backscatter'] / df['backscatter'].max()) * 100
    print("Generating plots...") #used gemini for visual things
    fig = plt.figure(figsize=(15, 12))
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(df['time_sec'], df['rssi'], 'b-', linewidth=1.5, alpha=0.8, label='RSSI (dBm)')
    colors = {'normal': 'lightgreen', 'metal': 'orange', 'touched': 'red'}
    df['event_change'] = df['event'].ne(df['event'].shift())
    change_indices = df[df['event_change']].index.tolist()
    
    for i in range(len(change_indices)):
        start_idx = change_indices[i]
        end_idx = change_indices[i+1] if i < len(change_indices)-1 else len(df)-1
        
        event_type = df.loc[start_idx, 'event']
        room_type = df.loc[start_idx, 'room_id']
        start_time = df.loc[start_idx, 'time_sec']
        end_time = df.loc[end_idx, 'time_sec']
        color = colors.get(event_type, 'gray')
        ax1.axvspan(start_time, end_time, alpha=0.2, color=color)
        mid_time = (start_time + end_time) / 2
        ax1.text(mid_time, df['rssi'].max() + 2, 
                 f"Room {room_type}\n{event_type.upper()}", 
                 ha='center', va='bottom', fontsize=9, fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

    ax1.set_ylabel('Received Signal Strength (dBm)', fontsize=11)
    ax1.set_title(f'Fig 1: Incoming Wi-Fi RSSI - {STUDENT_NAME}\n(Room and Event Transitions)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    ax1.set_xlim([0, df['time_sec'].max()])
    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(df['time_sec'], df['backscatter'], 'g-', linewidth=2, label='Backscatter Proxy (Std Dev)')
    ax2.axhline(y=30, color='gray', linestyle='--', alpha=0.5, label='Noise Floor')
    
    ax2.set_ylabel('Proxy Magnitude (Normalized)', fontsize=11)
    ax2.set_title('Fig 2: Backscattering Proxy Signal\n(Peaks indicate metal interaction/movement)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    ax2.set_xlim([0, df['time_sec'].max()])
    ax2.set_ylim([0, 110])
    ax3 = plt.subplot(3, 1, 3)
    rssi_norm = (df['rssi'] - df['rssi'].min()) / (df['rssi'].max() - df['rssi'].min()) * 100
    ax3.plot(df['time_sec'], rssi_norm, 'b-', alpha=0.6, linewidth=1, label='RSSI (Norm)')
    ax3.plot(df['time_sec'], df['backscatter'], 'r-', alpha=0.8, linewidth=1.5, label='Backscatter')
    ax3.set_xlabel('Time (seconds)', fontsize=11)
    ax3.set_ylabel('Normalized Amplitude (%)', fontsize=11)
    ax3.set_title('Fig 3: Signal Correlation Analysis', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right')
    ax3.set_xlim([0, df['time_sec'].max()])
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=300)
    print(f"Success! Plot saved to {OUTPUT_IMAGE}")
    plt.show()
if __name__ == "__main__":
    generate_plots()