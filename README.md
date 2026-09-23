# DBSCAN Customer Segmentation

## 📌 Description

This project implements a customer segmentation system using the **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** unsupervised machine learning algorithm.

Unlike supervised learning, no target variable is required. DBSCAN groups customers based on the density of their characteristics and identifies observations that do not belong to any sufficiently dense cluster as **noise**.

The trained model is integrated into an interactive **Streamlit web application** that allows users to enter customer characteristics and obtain a predicted cluster.

---

## 🎯 Objectives

The main objectives of this project are to:

- Apply an unsupervised learning approach to customer segmentation.
- Use DBSCAN to identify groups of similar customers.
- Detect potential noise or atypical observations.
- Save the trained model and its required artefacts.
- Build an interactive prediction interface with Streamlit.
- Deploy the application online.

---

## 🧠 Machine Learning Approach

### DBSCAN

DBSCAN stands for **Density-Based Spatial Clustering of Applications with Noise**.

The algorithm is based mainly on two parameters:

- `eps`: defines the radius of the neighborhood around each observation.
- `min_samples`: defines the minimum number of observations required to form a dense neighborhood.

DBSCAN classifies observations into:

- **Core points**: observations located in sufficiently dense regions.
- **Border points**: observations located close to a cluster but not dense enough to be core points.
- **Noise points**: observations that do not belong to any cluster.

Noise observations are represented by the label `-1`.

---

## ⚙️ Data Processing

Before applying DBSCAN, the customer variables are normalized.

This step is important because DBSCAN relies on distances between observations. Variables with larger scales could otherwise have a disproportionate influence on the clustering process.

For a new customer, the same normalization procedure is applied before calculating its distance to the learned core points.

---

## 🔎 Parameter Selection

Different combinations of `eps` and `min_samples` were evaluated.

The **Silhouette Score** was used as a criterion to compare the clustering configurations.

The selected DBSCAN configuration was then used to identify the core points and their associated cluster labels.

---

## 🚀 Prediction Strategy

Scikit-learn's DBSCAN implementation does not provide a native `predict()` method for assigning a new observation to an existing cluster.

Therefore, this project uses a custom assignment procedure based on the learned **core points**.

For a new customer:

1. The input variables are normalized.
2. The distances to the DBSCAN core points are calculated.
3. The closest core point is identified.
4. Its distance is compared with the DBSCAN `eps` value.
5. If the distance is within `eps`, the customer is assigned to the corresponding cluster.
6. Otherwise, the observation is classified as noise (`-1`).

This approach is used specifically to extend the DBSCAN clustering model to new observations.

---

## 🛠️ Technologies

The project was developed using:

- **Python**
- **NumPy**
- **Pandas**
- **Scikit-learn**
- **Joblib**
- **Streamlit**
- **Git / GitHub**

---

## 📂 Project Structure

```text
DBSCAN_Customer_Segmentation/
│
├── app.py
├── Customer_dataset.csv
├── modele_dbscan.joblib
├── requirements.txt
└── README.md
