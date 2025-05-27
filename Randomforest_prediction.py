import rasterio
import numpy as np
import os
from glob import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import joblib

# Paths
parameter_folder = "D:/Fire_Data/_Project_ForestFire_LCC/01_LCC_DATA/ModelTestData/CorrectedData_SSP126/"
label_raster_path = "D:/Fire_Data/_Project_ForestFire_LCC/01_LCC_DATA/ModelTestData/firelabel_raster/Forestfire_label_lmh1.tif"
output_prediction_path = "D:/Fire_Data/_Project_ForestFire_LCC/01_LCC_DATA/ModelTestData/output3/fireprone_prediction_ssp126_test1.tif"

# Load parameter rasters
parameter_files = sorted(glob(os.path.join(parameter_folder, "*.tif")))
print("✅ Found parameter files:", parameter_files)
if not parameter_files:
    raise ValueError("❌ No .tif files found in the given parameter folder.")

# Read and check raster shapes
parameters = []
shapes = []

for file in parameter_files:
    with rasterio.open(file) as src:
        data = src.read(1)
        shape = data.shape
        print(f"📂 Loaded: {os.path.basename(file)} with shape: {shape}")
        shapes.append((os.path.basename(file), shape))
        parameters.append(data)

# Check for mismatched shapes
first_shape = shapes[0][1]
mismatch_found = False
for fname, shape in shapes:
    if shape != first_shape:
        print(f"❌ Shape mismatch: {fname} has shape {shape}, expected {first_shape}")
        mismatch_found = True

if mismatch_found:
    raise ValueError("Shape mismatch detected among input rasters. Please align all raster layers.")

# Stack parameters
stacked_parameters = np.stack(parameters)

# Load label raster
with rasterio.open(label_raster_path) as src:
    labels = src.read(1)
    profile = src.profile
    label_nodata = src.nodata

# Fallback if NoData is undefined
if label_nodata is None:
    label_nodata = 255

# Flatten inputs
X_full = stacked_parameters.reshape(len(parameter_files), -1).T
y_full = labels.flatten()

# Build mask of valid labeled data
mask = (y_full != label_nodata) & (~np.isnan(y_full))
X_valid = X_full[mask]
y_valid = y_full[mask]

print(f"✅ Valid data points for training: {X_valid.shape[0]}")

# Handle NaNs in predictors
X_valid = np.nan_to_num(X_valid, nan=-9999)

# Downsample (optional)
sample_size = min(1000000, X_valid.shape[0])
X_sample, y_sample = resample(X_valid, y_valid, n_samples=sample_size, random_state=42, stratify=y_valid)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced", n_jobs=-1)
model.fit(X_train, y_train)
print("✅ Model trained successfully.")

# Save model
joblib.dump(model, "fire_prediction_model.pkl")

# ---- Evaluation ----
y_pred_test = model.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_test)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Classification report
print("🔍 Classification Report:")
print(classification_report(y_test, y_pred_test))

# Multiclass AUC (if applicable)
classes = np.unique(y_valid)
if len(classes) > 2:
    y_test_bin = label_binarize(y_test, classes=classes)
    y_pred_bin = label_binarize(y_pred_test, classes=classes)
    auc_score = roc_auc_score(y_test_bin, y_pred_bin, average="macro", multi_class="ovr")
    print(f"📈 Multiclass AUC (macro-averaged): {auc_score:.3f}")
else:
    y_prob = model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_prob)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()
    print(f"📈 Binary AUC: {auc_score:.3f}")

# ---- Predict on full dataset ----
X_full = np.nan_to_num(X_full, nan=-9999)
y_pred_full = np.full_like(y_full, label_nodata, dtype=np.uint8)
y_pred_full[mask] = model.predict(X_valid)

# Reshape to original raster
prediction = y_pred_full.reshape(labels.shape)

# Save prediction raster
profile.update(dtype=rasterio.uint8, count=1, nodata=label_nodata)
with rasterio.open(output_prediction_path, "w", **profile) as dst:
    dst.write(prediction, 1)

print("✅ Fire-prone areas predicted and saved to:", output_prediction_path)
