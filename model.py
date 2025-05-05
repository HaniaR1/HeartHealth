import pandas as pd
from sklearn.cluster import KMeans
import joblib

# Dummy data (you can replace with real dataset like UCI Heart Disease)
data = pd.DataFrame({
    'age': [29, 54, 45, 31, 62, 41],
    'bp': [130, 140, 120, 125, 160, 110],
    'cholesterol': [230, 250, 210, 240, 300, 190],
    'max_hr': [150, 135, 145, 160, 120, 170],
})

# KMeans with 2 clusters: likely heart disease (1) vs not (0)
model = KMeans(n_clusters=2, random_state=42)
model.fit(data)

# Save model
joblib.dump(model, 'model.pkl')
