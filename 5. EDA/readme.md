# 5. EDA
# Exploratory Data Analysis on Cardiographic Dataset

This repository contains an exploratory data analysis (EDA) of the `cardiographic.csv` dataset to uncover insights, identify patterns, and understand the underlying structure of fetal heart rate monitoring data.

---

## 📋 Table of Contents
- [Objective](#objective)
- [Dataset Overview](#dataset-overview)
- [Tools and Libraries](#tools-and-libraries)
- [Tasks](#tasks)
  - [Data Cleaning and Preparation](#data-cleaning-and-preparation)
  - [Statistical Summary](#statistical-summary)
  - [Data Visualization](#data-visualization)
  - [Pattern Recognition and Insights](#pattern-recognition-and-insights)
- [Conclusion](#conclusion)
- [Deliverables](#deliverables)

---

## Objective
The main goal of this assignment is to conduct a thorough exploratory analysis of the "cardiographic.csv" dataset using statistical summaries, visualizations, and data manipulation techniques.

---

## Dataset Overview
The dataset includes the following variables:
* **LB:** Baseline Fetal Heart Rate (FHR)
* **AC:** Accelerations
* **FM:** Fetal Movements
* **UC:** Uterine Contractions
* **DL:** Decelerations Late
* **DS:** Decelerations Short
* **DP:** Decelerations Prolonged
* **ASTV:** Percentage of Time with Abnormal Short Term Variability
* **MSTV:** Mean Value of Short Term Variability
* **ALTV:** Percentage of Time with Abnormal Long Term Variability
* **MLTV:** Mean Value of Long Term Variability

---

## Tools and Libraries
* **Language:** Python
* **Data Manipulation:** Pandas / NumPy
* **Data Visualization:** Matplotlib and Seaborn
* **Environment:** Jupyter Notebook

---

## Tasks

### Data Cleaning and Preparation
1. Load the dataset into a data frame.
2. Handle missing values appropriately (e.g., imputation, deletion).
3. Identify and correct data type inconsistencies.
4. Detect and treat outliers where necessary.

### Statistical Summary
* Provide measures of central tendency (mean, median) and dispersion (standard deviation, interquartile range) for each variable.
* Highlight interesting findings from the summary statistics.

### Data Visualization
* **Distributions:** Histograms and boxplots for numerical variables.
* **Categorical Variables:** Bar charts or pie charts for frequency distributions.
* **Relationships:** Scatter plots and correlation heatmaps.
* **Advanced Views:** Pair plots and violin plots for deeper structural insights.

### Pattern Recognition and Insights
* Identify correlations between variables and discuss their potential clinical/analytical implications.
* Look for trends or patterns over time if temporal data is available.

---

## Conclusion
* Summarize key insights and structural patterns discovered through the analysis.
* Discuss how findings impact decision-making or downstream modeling/analyses.

---

## Deliverables
* **Jupyter Notebook:** Complete file containing code, visualizations, and detailed explanations.
* **Summary Report:** Brief report outlining findings, insights, and recommendations.
