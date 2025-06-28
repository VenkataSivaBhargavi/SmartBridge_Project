import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
df = pd.read_csv(r"C:\Users\venka\Downloads\traffic volume (1).csv")
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], dayfirst=True)  
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
features = ['holiday', 'temp', 'rain', 'snow', 'weather', 'year', 'month', 'day']
target = 'traffic_volume'
missing = [col for col in features + [target] if col not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")
df = df.dropna(subset=features + [target])
label_encoders = {}
for col in ['holiday', 'weather']:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
X = df[features]
y = df[target]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = LinearRegression()
model.fit(X_scaled, y)
with open('scale.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
print("✅ Model, scaler, and encoders saved successfully.")