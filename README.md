# Indoor Localization using BLE RSSI Fingerprinting

## 📌 Project Overview

This project implements a basic **Indoor Localization System** based on **Bluetooth Low Energy (BLE) RSSI Fingerprinting**.

The system estimates a user’s indoor position by comparing real-time BLE signal strength (RSSI) measurements with a pre-collected 
fingerprint database using machine learning techniques (primarily K-Nearest Neighbors – KNN).

This work was done as part of an **internship project** for learning and demonstration purposes.

---

## 📂 Project Files

### Data Collection

* **BLEScanner.py** – Scans BLE devices and records RSSI values
* **ble_rssi_data_raw.csv** – Raw RSSI data collected from BLE beacons

### Data Processing

* **DataArranging.py** – Processes and arranges RSSI data
* **DataArrangingDir.py** – Data processing with direction information
* **ble_rssi_data_avg.csv** – Averaged RSSI fingerprint data
* **ble_rssi_data_avg_d.csv** – Averaged RSSI data with direction

### Localization

* **FindingBestK.py** – Finds optimal K value for KNN
* **PositionFinding.py** – Estimates position using RSSI fingerprinting
* **PositionFindingDir.py** – Position estimation with direction

### Analysis & Visualization

* **ErrorCalculation.xlsx** – Localization error calculation
* **Heatmap.xlsx** – Accuracy visualization using heatmap

### Presentation slides (Project progress)

* **RSSI Fingerprinting 1.pdf**
* **RSSI Fingerprinting 2.pdf**
* **RSSI Fingerprinting 3.pdf**

---

## 📊 Basic Workflow

1. Collect BLE RSSI data
2. Arrange and average RSSI values
3. Build fingerprint database
4. Apply KNN for position estimation
5. Analyze localization error

---

## Note

This project is for **learning and demonstration** as part of an **internship** and is not intended for production use.

