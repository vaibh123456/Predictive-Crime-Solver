import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump, load

np.random.seed(42)
n_samples = 1000
data = {
    'latitude': np.random.uniform(low=34.0, high=35.0, size=n_samples),
    'longitude': np.random.uniform(low=-119.0, high=-118.0, size=n_samples),
    'victim_age': np.random.randint(low=18, high=80, size=n_samples),
    'victim_gender': np.random.choice(['Male', 'Female'], size=n_samples),
    'victim_ethnicity': np.random.choice(['White', 'Black', 'Asian', 'Hispanic'], size=n_samples),
    'weather': np.random.choice(['Clear', 'Rainy', 'Cloudy'], size=n_samples),
    'day_night': np.random.choice(['Day', 'Night'], size=n_samples),
    'hour': np.random.randint(low=0, high=24, size=n_samples),
    'day_of_week': np.random.randint(low=0, high=7, size=n_samples),
    'month': np.random.randint(low=1, high=13, size=n_samples),
    'year': np.random.randint(low=2000, high=2024, size=n_samples),
    'crime_type': np.random.choice(['Theft', 'Assault', 'Burglary', 'Vandalism', 'Robbery'], size=n_samples)
}
df = pd.DataFrame(data)
df.to_csv('crime_data.csv', index=False)
print("Dataset generated and saved as 'crime_data.csv'")


data = pd.read_csv('crime_data.csv')
data['date_time'] = pd.to_datetime(data[['year', 'month', 'day_of_week']].assign(day=1))
data['hour'] = pd.to_datetime(data['hour'], format='%H').dt.hour
data['day_of_week'] = data['date_time'].dt.dayofweek
data['month'] = data['date_time'].dt.month
data['year'] = data['date_time'].dt.year
data = data.drop(columns=['date_time'])

label_encoders = {}
categorical_columns = ['victim_gender', 'victim_ethnicity', 'crime_type', 'weather', 'day_night']

for column in categorical_columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

X = data.drop(columns=['crime_type'])
y = data['crime_type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


dump(model, 'random_forest_model.joblib')
dump(scaler, 'scaler.joblib')

for column, le in label_encoders.items():
    dump(le, f'{column}_label_encoder.joblib')


label_encoders = {
    'victim_gender': load('victim_gender_label_encoder.joblib'),
    'victim_ethnicity': load('victim_ethnicity_label_encoder.joblib'),
    'weather': load('weather_label_encoder.joblib'),
    'day_night': load('day_night_label_encoder.joblib'),
    'crime_type': load('crime_type_label_encoder.joblib')
}


future_data = pd.DataFrame({
    'latitude': [34.0522],
    'longitude': [-118.2437],
    'victim_age': [30],
    'victim_gender': label_encoders['victim_gender'].transform(['Male']),
    'victim_ethnicity': label_encoders['victim_ethnicity'].transform(['Hispanic']),
    'weather': label_encoders['weather'].transform(['Clear']),
    'day_night': label_encoders['day_night'].transform(['Day']),
    'hour': [15],
    'day_of_week': [1],  
    'month': [3],
    'year': [2024]
})


model = load('random_forest_model.joblib')
scaler = load('scaler.joblib')

future_data = scaler.transform(future_data)
predicted_crime_type = model.predict(future_data)
predicted_crime_type = label_encoders['crime_type'].inverse_transform(predicted_crime_type)
print("Predicted Crime Type:", predicted_crime_type[0])
