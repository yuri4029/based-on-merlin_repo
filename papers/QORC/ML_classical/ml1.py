import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

# ----------------------------------------------------
# 1. LOAD THE DATA
# ----------------------------------------------------
# Load your labeled CSV file
df = pd.read_csv("labeled.csv")

# Assume the final column is named 'label'
# Ensure 'label' exists; change string to match your exact CSV column name if different
target_column = 'label' 

# ----------------------------------------------------
# 2. PREPROCESS TEXT COLUMNS (Encoding)
# ----------------------------------------------------
# NSL-KDD has categorical text columns like protocol_type, service, and flag
# Machine learning models only understand numbers, so we convert them
categorical_cols = df.select_dtypes(include=['object']).columns

# We loop through text columns except the target label itself
for col in categorical_cols:
    if col != target_column:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

# Convert target labels to simple binary: 0 for normal traffic, 1 for an attack
# (Alternatively, keep it as text for multi-class classification)
df[target_column] = df[target_column].apply(lambda x: 0 if str(x).strip().lower() == 'normal' else 1)

# ----------------------------------------------------
# 3. SPLIT INTO FEATURES AND TARGET
# ----------------------------------------------------
X = df.drop(columns=[target_column])
y = df[target_column]

# Split data: 80% for training the model, 20% for testing its accuracy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ----------------------------------------------------
# 4. SCALE THE NUMBERS
# ----------------------------------------------------
# Columns like 'src_bytes' have huge numbers, while 'count' has small numbers.
# We normalize them so they share the same scale.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ----------------------------------------------------
# 5. TRAIN THE MODEL
# ----------------------------------------------------
# We use a Decision Tree because it is fast, simple, and performs well on NSL-KDD
print("Training the Decision Tree model... Please wait.")
model = DecisionTreeClassifier(random_state=42, max_depth=10)
model.fit(X_train, y_train)

# ----------------------------------------------------
# 6. TEST AND EVALUATE
# ----------------------------------------------------
predictions = model.predict(X_test)

# Print out performance metrics
print("\n=== MODEL PERFORMANCE METRICS ===")
print(f"Overall Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
print("\nDetailed Classification Report:")
print(classification_report(y_test, predictions, target_names=["Normal (0)", "Attack (1)"]))
