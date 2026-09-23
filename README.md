# 🧩 DBSCAN Customer Segmentation

> An interactive customer segmentation application built with **DBSCAN**, an unsupervised machine learning algorithm based on density.

---

## 🚀 Live Demo

### 👉 Try the application online

🔗 **[DBSCAN Customer Segmentation — Live Demo](https://dbscancustomersegmentation-hpxjigd8whcru8iwwevwqh.streamlit.app/)**

The application allows you to enter a customer profile and interactively determine the DBSCAN cluster associated with that profile.

---

## 📌 About the Project

Customer segmentation is an important task in data science and marketing. It allows organizations to identify groups of customers with similar characteristics and better understand their profiles.

In this project, I implemented an **unsupervised learning approach using DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**.

The objective was not only to train a clustering model, but also to transform the model into a complete machine learning application:

**Data → Preprocessing → DBSCAN → Model Artefacts → New Customer → Prediction → Web Application → Deployment**

---

## 🎯 Project Objectives

The main objectives of this project are to:

- Apply an unsupervised learning algorithm to customer data.
- Identify groups of similar customers.
- Detect atypical observations considered as noise.
- Select suitable DBSCAN parameters.
- Normalize the data before clustering.
- Save the important model artefacts.
- Build an interactive prediction interface.
- Deploy the application online with Streamlit.

---

## 🧠 Why DBSCAN?

DBSCAN is a density-based clustering algorithm.

Unlike algorithms such as K-Means, DBSCAN does not require the number of clusters to be specified beforehand.

It identifies clusters according to the density of observations.

The algorithm mainly relies on two parameters:

### `eps`

Defines the radius of the neighborhood around an observation.

### `min_samples`

Defines the minimum number of observations required in this neighborhood for an observation to be considered a core point.

DBSCAN can identify three types of observations:

- 🟢 **Core points** — observations located in dense regions.
- 🔵 **Border points** — observations located close to a cluster.
- ⚪ **Noise points** — observations that do not belong to a sufficiently dense cluster.

In Scikit-learn, noise observations are represented by the label:

```text
-1
