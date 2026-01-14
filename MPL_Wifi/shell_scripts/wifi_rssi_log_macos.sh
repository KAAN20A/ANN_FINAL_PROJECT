#!/bin/bash

LOG_FILE="wifi_rssi_log.csv"

if [ ! -f "$LOG_FILE" ]; then
    echo "time_sec,rssi_pct,rssi_dBm" > "$LOG_FILE"
fi

START_TIME=$(date +%s)
DURATION=$((10 * 60))   

AIRPORT="/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport"

while [ $(( $(date +%s) - START_TIME )) -lt $DURATION ]; do

    RSSI_DBM=$($AIRPORT -I | grep "agrCtlRSSI" | awk '{print $2}')

    if [[ -z "$RSSI_DBM" ]]; then
        RSSI_DBM=0
        SIGNAL_PCT=0
    else
        
        SIGNAL_PCT=$(( (RSSI_DBM + 100) * 2 ))
        if [ "$SIGNAL_PCT" -gt 100 ]; then SIGNAL_PCT=100; fi
        if [ "$SIGNAL_PCT" -lt 0 ]; then SIGNAL_PCT=0; fi
    fi

    TIME_STAMP=$(date +"%H.%M.%S.%3N")

    echo "$TIME_STAMP,$SIGNAL_PCT,$RSSI_DBM" >> "$LOG_FILE"

    sleep 0.25
done

read -p "Press Enter to close..."
