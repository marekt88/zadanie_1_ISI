import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load data
print("Loading data...")
X_public = np.load('X_public.npy', allow_pickle=True)
y_public = np.load('y_public.npy', allow_pickle=True)
X_eval = np.load('X_eval.npy', allow_pickle=True)

# Print shapes
print("Shape of X_public:", X_public.shape)
print("Shape of y_public:", y_public.shape)
print("Shape of X_eval:", X_eval.shape)

# Convert numpy arrays to pandas DataFrames for easier handling
X_public_df = pd.DataFrame(X_public)
X_eval_df = pd.DataFrame(X_eval)

# Function to encode categorical variables
def encode_categorical(train_df, test_df):
    for col in train_df.columns:
        try:
            train_df[col] = pd.to_numeric(train_df[col])
            test_df[col] = pd.to_numeric(test_df[col])
        except:
            unique_vals = set(train_df[col].unique()) | set(test_df[col].unique())
            val_to_int = {val: i for i, val in enumerate(unique_vals)}
            train_df[col] = train_df[col].map(val_to_int)
            test_df[col] = test_df[col].map(val_to_int)
    return train_df, test_df

# Encode categorical variables
print("\nPreprocessing data...")
X_public_encoded, X_eval_encoded = encode_categorical(X_public_df.copy(), X_eval_df.copy())

# Convert to numpy arrays
X_public_processed = X_public_encoded.to_numpy()
X_eval_processed = X_eval_encoded.to_numpy()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_public_processed, y_public, test_size=0.2, random_state=42)

# Important feature indices (based on previous analysis)
important_features = [87, 68, 34, 42, 97, 31, 99, 18, 91, 78]

# Extract important features
X_train_important = X_train[:, important_features]
X_test_important = X_test[:, important_features]
X_eval_important = X_eval_processed[:, important_features]

# Create preprocessing pipeline for important features
print("\nCreating polynomial features for important features...")
important_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('poly', PolynomialFeatures(degree=2, include_bias=False))
])

# Transform important features
X_train_poly = important_pipeline.fit_transform(X_train_important)
X_test_poly = important_pipeline.transform(X_test_important)
X_eval_poly = important_pipeline.transform(X_eval_important)

# Create preprocessing pipeline for remaining features
remaining_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean'))
])

# Get remaining features
remaining_features = [i for i in range(X_train.shape[1]) if i not in important_features]
X_train_remaining = remaining_pipeline.fit_transform(X_train[:, remaining_features])
X_test_remaining = remaining_pipeline.transform(X_test[:, remaining_features])
X_eval_remaining = remaining_pipeline.transform(X_eval_processed[:, remaining_features])

# Combine all features
X_train_combined = np.hstack([X_train_remaining, X_train_poly])
X_test_combined = np.hstack([X_test_remaining, X_test_poly])
X_eval_combined = np.hstack([X_eval_remaining, X_eval_poly])

print(f"Shape after adding polynomial features - Training: {X_train_combined.shape}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_combined)
X_test_scaled = scaler.transform(X_test_combined)
X_eval_scaled = scaler.transform(X_eval_combined)

# Feature selection using LassoCV
print("\nPerforming feature selection...")
lasso = LassoCV(cv=5, random_state=42)
selector = SelectFromModel(lasso, prefit=False)

# Create final pipeline
pipeline = Pipeline([
    ('feature_selector', selector),
    ('regressor', LinearRegression())
])

# Train model
print("\nTraining model...")
pipeline.fit(X_train_scaled, y_train)

# Get selected feature mask
selected_features_mask = pipeline.named_steps['feature_selector'].get_support()
n_selected = sum(selected_features_mask)
print(f"\nNumber of features selected: {n_selected} out of {X_train_scaled.shape[1]}")

# Evaluate model
train_score = pipeline.score(X_train_scaled, y_train)
test_score = pipeline.score(X_test_scaled, y_test)

print("\nModel Performance:")
print(f"Training R² score: {train_score:.4f}")
print(f"Test R² score: {test_score:.4f}")

# Cross-validation score
cv_scores = cross_val_score(pipeline, X_train_scaled, y_train, cv=5, scoring='r2')
print(f"\nCross-validation R² scores: {cv_scores}")
print(f"Average CV R² score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Make predictions
y_eval_pred = pipeline.predict(X_eval_scaled)

# Save predictions
np.save('y_predikcia.npy', y_eval_pred)

# Verify predictions
loaded_pred = np.load('y_predikcia.npy')
print("\nPredictions saved successfully.")
print("Shape of predictions:", loaded_pred.shape)
print("First 5 predictions:", loaded_pred[:5])

# Get feature importance for selected features
feature_names = ([f"Original_{i}" for i in range(X_public.shape[1])] + 
                [f"Poly_{i}" for i in range(X_train_poly.shape[1])])
selected_features = np.array(feature_names)[selected_features_mask]
coefficients = pipeline.named_steps['regressor'].coef_[selected_features_mask]

importance_df = pd.DataFrame({
    'Feature': selected_features,
    'Coefficient': coefficients
})
importance_df['Abs_Coefficient'] = abs(importance_df['Coefficient'])
importance_df = importance_df.sort_values('Abs_Coefficient', ascending=False)

print("\nTop 10 most important features after selection:")
print(importance_df.head(10))

print("\nModel Summary:")
print("-" * 50)
print(f"Original features: {X_public.shape[1]}")
print(f"Features after polynomial expansion: {X_train_combined.shape[1]}")
print(f"Features after selection: {n_selected}")
print(f"Training R² score: {train_score:.4f}")
print(f"Test R² score: {test_score:.4f}")
print(f"Cross-validation mean R² score: {cv_scores.mean():.4f}")
print("-" * 50)
