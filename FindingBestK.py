import asyncio
import csv
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from bleak import BleakScanner

# กำหนดลิสต์ของ Device Address ที่ต้องการเก็บค่า RSSI
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

# กำหนดชื่อไฟล์
READ_FILENAME = 'ble_rssi_data_avg_d.csv'

# กำหนดเวลาการสแกนและจำนวนครั้งในการสแกน
scan_duration = 1  # วินาทีต่อการสแกนหนึ่งครั้ง
total_scans = 60  # จำนวนครั้งในการสแกนทั้งหมด

# ฟังก์ชันสำหรับสแกนอุปกรณ์ BLE
async def scan_ble_devices(no_of_scans, duration):
    rssi_data = defaultdict(list)
    
    # ทำการสแกนทั้งหมด no_of_scans ครั้ง
    for _ in range(no_of_scans):
        devices = await BleakScanner.discover(timeout=duration)
        for device in devices:
            if device.address in TARGET_DEVICES:
                rssi_data[device.address].append(device.rssi)
    
    # คำนวณค่าเฉลี่ยของ RSSI สำหรับแต่ละ device โดยตัดค่านอกเกณฑ์ออก
    avg_rssi = {}
    for address, rssi in rssi_data.items():
        if rssi:
            q1 = np.percentile(rssi, 25)
            q3 = np.percentile(rssi, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            filtered_rssi = [value for value in rssi if lower_bound <= value <= upper_bound]
            if filtered_rssi:
                avg_rssi[address] = int(np.mean(filtered_rssi))
            else:
                avg_rssi[address] = int(np.mean(rssi))
    
    return avg_rssi

# ฟังก์ชันสำหรับอ่านค่า RSSI จากไฟล์ CSV
def read_rssi_from_csv(filename):
    data = []
    with open(filename, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    return data

# ฟังก์ชันสำหรับหาตำแหน่งที่ใกล้ที่สุด
def find_closest_match(avg_rssi, csv_data, k):
    closest_matches = []

    for entry in csv_data:
        # กำหนดตัวแปรเก็บค่าผลต่างรวม
        total_diff = 0
        
        # วนลูปอุปกรณ์ที่มีค่า RSSI ใน avg_rssi
        for device, avg_rssi_value in avg_rssi.items():
            # ตรวจสอบว่ามีค่า RSSI เฉลี่ยและข้อมูลของอุปกรณ์ใน entry
            if avg_rssi_value is not None and entry[device]:
                # คำนวณค่าผลต่างรวมของ RSSI ระหว่างค่าเฉลี่ยและข้อมูลใน entry
                total_diff += abs(avg_rssi_value - int(entry[device]))
        
        closest_matches.append(((entry['x'], entry['y']), total_diff))

    # เรียงลำดับ closest_matches ตามค่า total_diff จากน้อยไปมาก และคืนค่า k ลำดับแรก
    closest_matches.sort(key=lambda x: x[1])
    
    return closest_matches[:k]

# ฟังก์ชันสำหรับการคำนวณหาจุดกึ่งกลาง (KNN)
def calculate_knn_average_position(closest_matches):
    sum_x, sum_y = 0, 0
    for (x, y), _ in closest_matches:
        sum_x += int(x)
        sum_y += int(y)
    
    avg_x = round(sum_x / len(closest_matches))
    avg_y = round(sum_y / len(closest_matches))

    return avg_x, avg_y

# ฟังก์ชันสำหรับการคำนวณหาจุดกึ่งกลาง (WKNN)
def calculate_weighted_average_position(closest_matches):
    sum_x, sum_y, total_weight = 0, 0, 0
    for (x, y), diff in closest_matches:
        weight = 1/diff
        sum_x += int(x) * weight
        sum_y += int(y) * weight
        total_weight += weight
        
    if total_weight > 0:
        avg_x = round(sum_x / total_weight)
        avg_y = round(sum_y / total_weight)
    else:
        avg_x, avg_y = 0, 0
        
    return avg_x, avg_y

# ฟังก์ชันสำหรับคำนวณ mean error และ SD error
def calculate_errors(estimated_positions, true_positions):
    errors = np.linalg.norm(np.array(estimated_positions) - np.array(true_positions), axis=1)
    mean_error = np.mean(errors)
    #sd_error = np.std(errors)
    return mean_error

async def main():
    # อ่านค่า RSSI จากไฟล์ CSV
    csv_data = read_rssi_from_csv(READ_FILENAME)
    
    # รับค่าตำแหน่งจริงจากคีย์บอร์ด
    true_x = float(input("Enter true x position: "))
    true_y = float(input("Enter true y position: "))
    true_position = (true_x, true_y)
    
    print(f"Scanning for {scan_duration * total_scans} seconds in total...")
    
    # ทำการสแกนอุปกรณ์ BLE
    avg_rssi = await scan_ble_devices(total_scans, scan_duration)    
    
    # สำหรับเก็บค่า mean error ของทั้ง KNN และ WKNN สำหรับทุกค่า k ตั้งแต่ 1 ถึง 33
    knn_mean_errors = []
    wknn_mean_errors = []
    ks = list(range(1, 34)) # 34 & 133
    
    # ทำการหาค่า mean error สำหรับทุกค่า k
    for k in ks:
        closest_matches = find_closest_match(avg_rssi, csv_data, k)
        if k == max(ks):
            print(closest_matches)
        
        # คำนวณจุดกึ่งกลาง (KNN)
        knn_estimated_position = calculate_knn_average_position(closest_matches)
        
        # คำนวณ mean error สำหรับ KNN
        knn_mean_error = calculate_errors([knn_estimated_position], [true_position])
        knn_mean_errors.append(knn_mean_error)
        
        # คำนวณจุดกึ่งกลาง (WKNN)
        wknn_x, wknn_y = calculate_weighted_average_position(closest_matches)
        
        # คำนวณ mean error สำหรับ WKNN
        wknn_mean_error = calculate_errors([wknn_x, wknn_y], [true_position])
        wknn_mean_errors.append(wknn_mean_error)

    # พล็อตกราฟ mean error สำหรับทั้ง KNN และ WKNN ตามค่า k
    plt.figure(figsize=(10, 6))
    plt.plot(ks, knn_mean_errors, marker='o', linestyle='-', color='b', label='KNN')
    plt.plot(ks, wknn_mean_errors, marker='o', linestyle='-', color='g', label='WKNN')
    plt.title('Mean Error vs. k for KNN and WKNN')
    plt.xlabel('k')
    plt.ylabel('Mean Error')
    plt.xticks(ks)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# เริ่มต้นการทำงานของโปรแกรม
if __name__ == "__main__":
    asyncio.run(main())
