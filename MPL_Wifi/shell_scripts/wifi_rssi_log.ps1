param(
    [int]$durationMinutes = 1  
)


$logFile = "wifi_rssi_log.csv"


if (!(Test-Path $logFile)) {
    "time_sec,rssi_pct,rssi_dBm" | Out-File $logFile
}
$startTime = Get-Date


while ((Get-Date) -lt $startTime.AddMinutes($durationMinutes)) {
    
    $output = netsh wlan show interfaces

    
    $signalLine = ($output | Select-String "Signal").ToString()
    if ($signalLine -match "(\d+)%") {
        $signalPercent = [int]$Matches[1]
        
        $rssiDbm = ($signalPercent / 2) - 100
    } else {
        $signalPercent = 0
        $rssiDbm = 0
    }
      Write-Output "$rssiDbm,$rssiDbm,$rssiDbm,$rssiDbm"
     
    
    $time = (Get-Date).ToString("HH.mm.ss.fff")
    
    
    "$time,$signalPercent,$rssiDbm" | Out-File $logFile -Append

    
    Start-Sleep -Milliseconds 250
}


