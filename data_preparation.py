import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the dataset
data = pd.read_csv('crime_data.csv')

# Feature engineering: Extract date and time features
data['date_time'] = pd.to_datetime(data['date_time'])
data['hour'] = data['date_time'].dt.hour
data['day_of_week'] = data['date_time'].dt.dayofweek
data['month'] = data['date_time'].dt.month
data['year'] = data['date_time'].dt.year

# Drop the original date_time column
data = data.drop(columns=['date_time'])

# Encoding categorical features
label_encoders = {}
categorical_columns = ['victim_gender', 'victim_ethnicity', 'crime_type', 'weather', 'day_night']

for column in categorical_columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Define features and target variable
X = data.drop(columns=['crime_type'])  # Use other columns as features
y = data['crime_type']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
