import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import joblib

# Load the dataset
df = pd.read_csv('antimicrobial_resistance_data_large.csv')

# Drop unnecessary columns
df = df.drop(columns=['location', 'collection_date'])

# Encode categorical variables
label_encoders = {}
for column in ['pathogen', 'antibiotic']:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Split the data into features and target
X = df.drop(columns=['resistance_level'])
y = df['resistance_level']

# Encode the target labels
y = LabelEncoder().fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define individual models
pipe_rf = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier())
])

pipe_gb = Pipeline([
    ('scaler', StandardScaler()),
    ('model', GradientBoostingClassifier())
])

pipe_xgb = Pipeline([
    ('scaler', StandardScaler()),
    ('model', XGBClassifier())
])

# Hyperparameter tuning
param_grid = {
    'model__n_estimators': [50, 100, 200],
    'model__max_depth': [None, 10, 20, 30]
}

gs_rf = GridSearchCV(pipe_rf, param_grid, cv=3, n_jobs=-1)
gs_gb = GridSearchCV(pipe_gb, param_grid, cv=3, n_jobs=-1)
gs_xgb = GridSearchCV(pipe_xgb, param_grid, cv=3, n_jobs=-1)

# Fit the models
gs_rf.fit(X_train, y_train)
gs_gb.fit(X_train, y_train)
gs_xgb.fit(X_train, y_train)

# Combine the models into an ensemble
ensemble_model = VotingClassifier(estimators=[
    ('rf', gs_rf.best_estimator_),
    ('gb', gs_gb.best_estimator_),
    ('xgb', gs_xgb.best_estimator_)
], voting='hard')

# Train the ensemble model
ensemble_model.fit(X_train, y_train)

# Save the model
joblib.dump(ensemble_model, 'ensemble_model.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')

# Evaluate the model
y_pred = ensemble_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High'])

print("Accuracy:", accuracy)
print("Classification Report:\n", report)
