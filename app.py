# -*- coding: utf-8 -*-
"""
Created on May 9, 2025
@author: Phattharajin Joyjaroen
"""

import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Page title
st.title("🍷 K-Means Clustering on Wine Quality Dataset by Phattharajin Joyjaroen")

# Load dataset
@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    df = pd.read_csv(url, sep=';')
    return df

df = load_data()
X = df.drop(columns=["quality"])  # Remove the label column

# Sidebar - Number of clusters
st.sidebar.header("🔧 Configure Clustering")
k = st.sidebar.slider("Select number of clusters (k)", 2, 10, 3)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Run K-Means
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X_scaled)
labels = kmeans.labels_

# Dimensionality reduction with PCA
pca = PCA(n_components=2)
reduced = pca.fit_transform(X_scaled)
reduced_df = pd.DataFrame(reduced, columns=["PCA1", "PCA2"])
reduced_df["Cluster"] = labels

# Plot clusters
fig, ax = plt.subplots()
for cluster in range(k):
    cluster_data = reduced_df[reduced_df["Cluster"] == cluster]
    ax.scatter(cluster_data["PCA1"], cluster_data["PCA2"], label=f"Cluster {cluster}")
ax.set_title("Clusters of Wine Data (2D PCA Projection)")
ax.set_xlabel("PCA1")
ax.set_ylabel("PCA2")
ax.legend()

# Show plot and data
st.pyplot(fig)
st.markdown("### 📊 First 10 Entries of Clustered PCA Data")
st.dataframe(reduced_df.head(10))
