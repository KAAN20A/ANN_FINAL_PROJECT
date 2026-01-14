#!/bin/bash

LOG_FILE="wifi_rssi_log.csv"


if [ ! -f "$LOG_FILE" ]; then
    echo "time_sec,rssi_pct,rssi_dBm" > "$LOG_FILE"
fi

START_TIME=$(date +%s)
DURATION=$((10 * 60))   

while [ $(( $(date +%s) - START_TIME )) -lt $DURATION ]; do

    
    SIGNAL_PCT=$(nmcli -t -f IN-USE,SIGNAL dev wifi list | grep '^*' | cut -d: -f2)

    if [[ -z "$SIGNAL_PCT" ]]; then
        SIGNAL_PCT=0
        RSSI_DBM=0
    else
        
        RSSI_DBM=$(( SIGNAL_PCT / 2 - 100 ))
    fi

    TIME_STAMP=$(date +"%H.%M.%S.%3N")

   
    echo "$TIME_STAMP,$SIGNAL_PCT,$RSSI_DBM" >> "$LOG_FILE"

   
    sleep 0.25
done

read -p "Press Enter to close..."
