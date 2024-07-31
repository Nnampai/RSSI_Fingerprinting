import csv
from collections import defaultdict
import numpy as np

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
    "FC:92:D5:14:99:90"
]

READ_FILENAME = 'ble_rssi_data_raw.csv'
WRITE_FILENAME = 'ble_rssi_data_avg.csv'

def read_rssi_data(filename):
    data = []
    with open(filename, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    return data

def calculate_average_rssi(data, target_devices):
    rssi_values = defaultdict(lambda: {device: [] for device in target_devices})
    # เก็บค่า RSSI ที่มี x และ y เดียวกัน
    for entry in data:
        x = entry['x']
        y = entry['y']
        key = (x, y)
        
        for device in target_devices:
            if entry[device]:
                rssi_values[key][device].append(int(entry[device]))
    
    # หาค่า RSSI เฉลี่ยโดยตัดค่า outlier ออก
    averages = []
    for key in rssi_values:
        x, y = key
        avg_entry = {'x': x, 'y': y}
        for device in target_devices:
            values = rssi_values[key][device]
            if values:
                q1 = np.percentile(values, 25)
                q3 = np.percentile(values, 75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                filtered_values = [v for v in values if lower_bound <= v <= upper_bound]
                if filtered_values:
                    avg_entry[device] = round(np.mean(filtered_values))
                else:
                    avg_entry[device] = ''
            else:
                avg_entry[device] = ''
        averages.append(avg_entry)
    
    return averages

# ฟังก์ชันสำหรับบันทึกข้อมูล RSSI เฉลี่ยลงในไฟล์ CSV
def write_average_rssi_to_csv(averages, target_devices, filename):
    # จัดเรียงข้อมูลตาม y และ x
    sorted_averages = sorted(averages, key=lambda entry: (int(entry['y']), int(entry['x'])))
    
    # เขียนข้อมูลลงในไฟล์ CSV
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ["No", "x", "y"] + target_devices
        writer.writerow(header)
        
        for index, avg_entry in enumerate(sorted_averages, start=1):
            row = [index, avg_entry['x'], avg_entry['y']]
            for device in target_devices:
                row.append(avg_entry[device])
            writer.writerow(row)

def main():
    data = read_rssi_data(READ_FILENAME)
    averages = calculate_average_rssi(data, TARGET_DEVICES)
    write_average_rssi_to_csv(averages, TARGET_DEVICES, WRITE_FILENAME)
    print(f"RSSI average data saved to '{WRITE_FILENAME}'.")

# เริ่มต้นการทำงานของโปรแกรม
if __name__ == "__main__":
    main()
