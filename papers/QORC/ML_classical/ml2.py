import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt

# 1) Load data
csv_path = "NSL_KDD_labeled.csv"  # adjust path as needed
df = pd.read_csv(csv_path)

categorical_cols = ["protocol_type", "service", "flag"]
target_col = "label"
feature_cols = [c for c in df.columns if c != target_col]
numeric_cols = [c for c in feature_cols if c not in categorical_cols]

# 2) Binary target: normal vs attack (set to False to keep multiclass attack names)
BINARY_TASK = True
if BINARY_TASK:
    y_raw = np.where(df[target_col] == "normal", "normal", "attack")
else:
    y_raw = df[target_col]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

X = df[feature_cols]

# 3) Train/test split (stratified to keep class proportions)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4) Preprocessing: one-hot encode categoricals, scale numeric features
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", StandardScaler(), numeric_cols),
    ]
)

# 5) Model: RandomForest is a strong, simple baseline for tabular IDS data
model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("clf", RandomForestClassifier(
        n_estimators=200, max_depth=None, random_state=42, n_jobs=-1
    )),
])

# 6) Train
model.fit(X_train, y_train)

# 7) Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=label_encoder.classes_).plot(
    cmap="Blues", xticks_rotation=45
)
plt.tight_layout()
plt.show()

# 8) Feature importances (top 15)
feature_names = model.named_steps["preprocess"].get_feature_names_out()
importances = model.named_steps["clf"].feature_importances_
top_idx = np.argsort(importances)[::-1][:15]
for idx in top_idx:
    print(f"{feature_names[idx]}: {importances[idx]:.4f}")