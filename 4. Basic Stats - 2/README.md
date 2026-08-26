# 4. BASIC STATS - 2

# Problem Statement: Hospital Patient Data Analysis

## Context
A hospital maintains patient records including admission details, department, diagnosis, doctor, and bill amount. The goal of this analysis is to process patient and billing datasets, clean missing records, handle duplicates, and prepare a unified DataFrame for department-wise and doctor-performance analytics.

---

## Tasks Completed
* **Data Inspection**: Loaded the patient dataset and examined its structural summary using `info()`.
* **Feature Selection**: Selected key billing-related columns: `['PatientID', 'Department', 'Doctor', 'BillAmount']`.
* **Data Cleaning**: Dropped administrative columns (`['ReceptionistID', 'CheckInTime']`) to focus purely on analytical metrics.
* **Missing Value Imputation**: Calculated global mean bill amounts and filled missing `BillAmount` records.
* **Deduplication**: Removed duplicate records based on `PatientID` to handle multiple follow-up entries cleanly.
* **Aggregation**: Calculated total revenue generated per department using `groupby()`.
* **Data Integration**: Merged patient information with external billing datasets on `PatientID`.
* **Concatenation (Row & Column)**:
  * Append new patient records row-wise for current week updates.
  * Joined new financial metric columns column-wise (`['InsuranceCovered', 'FinalAmount']`).

---

## Expected Outcomes
* **Cleaned Dataset**: Fully preprocessed dataset free of missing values and duplicate rows.
* **Integrated Records**: Unified data model linked across `PatientID`.
* **Analytics Readiness**: Ready for downstream analysis on department-wise revenue, doctor billing performance, and financial distribution.

---

## Technical Stack
* **Language**: Python
* **Libraries**: Pandas, NumPy
* **Environment**: Jupyter Notebook / Google Colab
