import asyncio
from bleak import BleakScanner
import csv
from datetime import datetime
import sys

TARGET_DEVICES = [
    "E5:53:39:5F:31:1A",
    "F8:9D:4B:5D:44:F4",
    "FD:AF:34:29:05:FE",
    "C6:62:EC:2D:90:AB",
    "D6:77:D9:C0:A6:54",
    "FA:C4:4A:07:18:9F",
    "C4:35:CE:FD:FB:DF",
    "CC:7F:D5:E0:41:92",
    "D7:B8:72:01:49:AC",
    "C5:EB:92:59:4D:1A",
    "C2:80:A9:28:D0:79",
    "FC:92:D5:14:99:90",
]

FILENAME = '/home/pi/RSSI_Fingerprinting/ble_rssi_data_raw.csv'

scan_duration = 1
total_scans = 60

async def scan_ble_devices(duration):
    devices = await BleakScanner.discover(timeout=duration)
    return devices

def save_rssi_to_csv(rssi_data, target_devices, filename):
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Table Header (Only first time)
        if file.tell() == 0:
            header = ["time", "x", "y", "direction"]
            header.extend(target_devices)
            writer.writerow(header)
            
        for row in rssi_data:
            writer.writerow(row)
            
async def main():
    # From command line
    if len(sys.argv) != 4:
        print("Usage: python3 script_name.py <x> <y> <direction>")
        sys.exit(1)
    
    x = int(sys.argv[1])
    y = int(sys.argv[2])
    direction = sys.argv[3]
    
    rssi_data = []
    
    print(f"Scanning for {scan_duration * total_scans} seconds in total...")
    
    for _ in range(total_scans):
        devices = await scan_ble_devices(scan_duration)
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        row = [current_time, x, y, direction]
        
        device_rssi_map = {dev.address: dev.rssi for dev in devices if dev.address in TARGET_DEVICES}
        
        for dev in TARGET_DEVICES:
            row.append(device_rssi_map.get(dev,""))
        rssi_data.append(row)
        
    save_rssi_to_csv(rssi_data, TARGET_DEVICES, FILENAME)
    print(f"RSSI data saved to '{FILENAME}'.")
    
if __name__ == "__main__":
    asyncio.run(main())