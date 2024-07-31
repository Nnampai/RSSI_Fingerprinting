import asyncio
from bleak import BleakScanner
import csv
from collections import defaultdict
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

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
READ_FILENAME = 'ble_rssi_data_avg.csv'

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
def find_closest_match(avg_rssi, csv_data, k=4):
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
        avg_x, avg_y = 0, 0  # หรือค่าเริ่มต้นอื่นๆที่เหมาะสม

    return avg_x, avg_y

# ฟังก์ชันสำหรับคำนวณ mean error
def calculate_errors(estimated_positions, true_positions):
    errors = np.linalg.norm(np.array(estimated_positions) - np.array(true_positions), axis=1)
    mean_error = np.mean(errors)
    return mean_error

# ฟังก์ชันสำหรับการฝึกและทำนายตำแหน่งโดยใช้ SVM และ SVR
def train_and_predict_svm(csv_data, avg_rssi):
    # เตรียมข้อมูลสำหรับการฝึก
    X = []
    y_x = []
    y_y = []
    
    for entry in csv_data:
        rssi_values = [int(entry[device]) if entry[device] else -100 for device in TARGET_DEVICES]  # เติมค่า -100 สำหรับค่า RSSI ที่ขาดหายไป
        X.append(rssi_values)
        y_x.append(float(entry['x']))
        y_y.append(float(entry['y']))
    
    # Normalize data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # สร้างและฝึกโมเดล SVR สำหรับ x และ y โดยใช้ GridSearchCV
    svr_x = GridSearchCV(SVR(), param_grid={"C": [0.1, 1, 10], "epsilon": [0.1, 0.2, 0.5], "gamma": ["scale", "auto"]}, cv=5)
    svr_y = GridSearchCV(SVR(), param_grid={"C": [0.1, 1, 10], "epsilon": [0.1, 0.2, 0.5], "gamma": ["scale", "auto"]}, cv=5)
    
    svr_x.fit(X, y_x)
    svr_y.fit(X, y_y)
    
    # เตรียมข้อมูลสำหรับการทำนาย
    test_rssi = [avg_rssi.get(device, -100) for device in TARGET_DEVICES]
    test_rssi = scaler.transform([test_rssi])
    
    # ทำนายตำแหน่ง
    predicted_x = round(svr_x.predict(test_rssi)[0])
    predicted_y = round(svr_y.predict(test_rssi)[0])
    
    return predicted_x, predicted_y

async def main():
    # อ่านค่า RSSI จากไฟล์ CSV
    csv_data = read_rssi_from_csv(READ_FILENAME)
    
    # รับค่าตำแหน่งจริงจากคีย์บอร์ด
    true_x = float(input("Enter true x position: "))
    true_y = float(input("Enter true y position: "))
    true_position = (true_x, true_y)

    print(f"Finding position in {scan_duration * total_scans} seconds...")
    
    # ทำการสแกนอุปกรณ์ BLE
    avg_rssi = await scan_ble_devices(total_scans, scan_duration)
    
    # ค้นหา x และ y ที่มีค่า RSSI ใกล้เคียงที่สุด
    k = 3
    closest_matches = find_closest_match(avg_rssi, csv_data, k)
    print(closest_matches)

    # คำนวณจุดกึ่งกลาง (KNN)
    knn_estimated_position = calculate_knn_average_position(closest_matches)
    print(f"The KNN estimated position is x={knn_estimated_position[0]}, y={knn_estimated_position[1]}")

    # คำนวณจุดกึ่งกลาง (WKNN)
    wknn_estimated_position = calculate_weighted_average_position(closest_matches)
    print(f"The WKNN estimated position is x={wknn_estimated_position[0]}, y={wknn_estimated_position[1]}")

    # คำนวณ mean error สำหรับ KNN
    knn_mean_error = calculate_errors([knn_estimated_position], [true_position])
    print(f"KNN Mean Error: {knn_mean_error}")

    # คำนวณ mean error สำหรับ WKNN
    wknn_mean_error = calculate_errors([wknn_estimated_position], [true_position])
    print(f"WKNN Mean Error: {wknn_mean_error}")
    
    # ฝึกและทำนายตำแหน่งโดยใช้ SVM
    svm_predicted_x, svm_predicted_y = train_and_predict_svm(csv_data, avg_rssi)
    print(f"SVM predicted position is x={svm_predicted_x}, y={svm_predicted_y}")

    # คำนวณ mean error สำหรับ SVM
    svm_mean_error = calculate_errors([(svm_predicted_x, svm_predicted_y)], [true_position])
    print(f"SVM Mean Error: {svm_mean_error}")

# เริ่มต้นการทำงานของโปรแกรม
if __name__ == "__main__":
    asyncio.run(main())
