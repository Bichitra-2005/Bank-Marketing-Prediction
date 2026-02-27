import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# 1. Load data
try:
    # sep=None with engine='python' tells pandas to guess the separator (comma or semicolon)
    df = pd.read_csv('bank.csv', sep=None, engine='python')
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit()

# Clean column names (removes quotes or hidden spaces)
df.columns = df.columns.str.strip().str.replace('"', '').str.replace("'", "")
print("Columns found:", df.columns.tolist())

# 2. DROP DURATION - The 'leaky' feature that makes it always predict 'yes'
if 'duration' in df.columns:
    df = df.drop(columns=['duration'])
    print("Successfully dropped 'duration'.")

# 3. SET TARGET COLUMN
# Based on your terminal output, your target is named 'deposit'
target_col = 'deposit' 

if target_col not in df.columns:
    print(f"Error: Target column '{target_col}' not found. Please check your CSV column names.")
    exit()

# 4. Preprocessing
X = df.drop(columns=[target_col])
# Convert 'yes'/'no' in deposit column to 1/0
y = df[target_col].str.strip().str.replace('"', '').apply(lambda x: 1 if x == 'yes' else 0)

# One-hot encoding for categorical variables (job, marital, etc.)
X = pd.get_dummies(X)

# 5. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Scaling (Crucial for Logistic Regression performance)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. Train with Class Weight balancing to prevent "Always Yes" bias
model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train_scaled, y_train)

# 8. Save the brain, the scale, and the feature list
joblib.dump(model, 'bank_model_new.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(list(X.columns), 'model_columns.pkl')

print("\n✅ Success! Model trained and saved.")
print(f"Accuracy on test set: {model.score(X_test_scaled, y_test):.2f}")