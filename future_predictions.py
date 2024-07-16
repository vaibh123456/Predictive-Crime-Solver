import pandas as pd
from joblib import load


model = load('random_forest_model.joblib')
scaler = load('scaler.joblib')


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


future_data = scaler.transform(future_data)


predicted_crime_type = model.predict(future_data)
predicted_crime_type = label_encoders['crime_type'].inverse_transform(predicted_crime_type)

print("Predicted Crime Type:", predicted_crime_type[0])
