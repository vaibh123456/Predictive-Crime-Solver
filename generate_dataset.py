import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_high_quality_data(num_records):
    np.random.seed(42)
    
    
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 12, 31)
    date_range = (end_date - start_date).days
    date_times = [start_date + timedelta(days=np.random.randint(date_range)) for _ in range(num_records)]
    
    
    latitudes = np.random.uniform(low=34.0, high=35.0, size=num_records)
    longitudes = np.random.uniform(low=-119.0, high=-118.0, size=num_records)
    
    
    victim_ages = np.random.randint(18, 80, size=num_records)
    victim_genders = np.random.choice(['Male', 'Female'], size=num_records)
    victim_ethnicities = np.random.choice(['Hispanic', 'White', 'Black', 'Asian', 'Other'], size=num_records)
    
    
    weather_conditions = np.random.choice(['Clear', 'Rainy', 'Foggy', 'Cloudy'], size=num_records)
    day_night = np.random.choice(['Day', 'Night'], size=num_records)
    
    
    crime_types = np.random.choice(['Theft', 'Assault', 'Robbery', 'Burglary', 'Vandalism'], size=num_records, p=[0.3, 0.2, 0.15, 0.15, 0.2])
    
    
    df = pd.DataFrame({
        'date_time': date_times,
        'latitude': latitudes,
        'longitude': longitudes,
        'victim_age': victim_ages,
        'victim_gender': victim_genders,
        'victim_ethnicity': victim_ethnicities,
        'weather': weather_conditions,
        'day_night': day_night,
        'hour': [dt.hour for dt in date_times],
        'day_of_week': [dt.weekday() for dt in date_times],
        'month': [dt.month for dt in date_times],
        'year': [dt.year for dt in date_times],
        'crime_type': crime_types
    })
    
    return df


num_records = 2000  
df = generate_high_quality_data(num_records)
df.to_csv('high_quality_crime_data.csv', index=False)

print("High-quality dataset created and saved as 'high_quality_crime_data.csv'.")
