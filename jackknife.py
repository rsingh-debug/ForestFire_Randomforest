import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import rasterio
import os
from glob import glob

# Load and stack all parameter rasters
parameter_folder = "D:/Fire_Data/_Project_ForestFire_LCC/01_LCC_DATA/ModelTestData/Jackknife_data/"
label_path = "D:/Fire_Data/_Project_ForestFire_LCC/01_LCC_DATA/ModelTestData/Jackknife_label_raster/ff_label_lmh.tif"
parameter_files = sorted(glob(os.path.join(parameter_folder, "*.tif")))

param_names = [os.path.basename(p).replace(".tif", "") for p in parameter_files]
print("Found features:", param_names)

# Load all rasters
layers = []
for path in parameter_files:
    with rasterio.open(path) as src:
        layers.append(src.read(1).flatten())

X = np.stack(layers, axis=1)

# Load labels
with rasterio.open(label_path) as src:
    y = src.read(1).flatten()
    label_nodata = src.nodata

# Mask nodata
mask = (y != label_nodata) & (~np.isnan(y))
X = X[mask]
y = y[mask]

# Replace NaNs in features
X = np.nan_to_num(X, nan=-9999)

# Split train/test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)

# Train full model
full_model = RandomForestClassifier(n_estimators=100, random_state=42)
full_model.fit(X_train, y_train)
y_pred_full = full_model.predict(X_test)
acc_full = accuracy_score(y_test, y_pred_full)
print("Full model accuracy:", acc_full)

# Jackknife: leave one parameter out
acc_jk_list = []
for i in range(X.shape[1]):
    print(f"Testing without feature: {param_names[i]}")
    
    X_train_jk = np.delete(X_train, i, axis=1)
    X_test_jk = np.delete(X_test, i, axis=1)

    model_jk = RandomForestClassifier(n_estimators=100, random_state=42)
    model_jk.fit(X_train_jk, y_train)

    y_pred_jk = model_jk.predict(X_test_jk)
    acc_jk = accuracy_score(y_test, y_pred_jk)

    acc_jk_list.append(acc_jk)

# Plot
import matplotlib.pyplot as plt

importance_drop = [acc_full - acc for acc in acc_jk_list]
plt.figure(figsize=(12, 6))
plt.barh(param_names, importance_drop, color="tomato")
plt.xlabel("Decrease in Accuracy when Feature is Removed")
plt.title("Jackknife Analysis - Feature Contribution")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
