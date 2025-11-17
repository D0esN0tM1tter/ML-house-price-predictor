import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pickle

# Génération du dataset synthétique
np.random.seed(42)
n_samples = 1000

data = {
    'square_feet': np.random.randint(800, 4000, n_samples),
    'bedrooms': np.random.randint(1, 6, n_samples),
    'bathrooms': np.random.randint(1, 4, n_samples),
    'age_years': np.random.randint(0, 50, n_samples),
    'lot_size': np.random.randint(2000, 10000, n_samples),
    'garage_spaces': np.random.randint(0, 3, n_samples),
    'neighborhood_score': np.random.randint(1, 11, n_samples)
}

df = pd.DataFrame(data)

# Formule de prix
df['price'] = (
    100000 +  # Prix de base
    (df['square_feet'] * 150) +
    (df['bedrooms'] * 20000) +
    (df['bathrooms'] * 15000) -
    (df['age_years'] * 2000) +
    (df['lot_size'] * 10) +
    (df['garage_spaces'] * 10000) +
    (df['neighborhood_score'] * 5000) +
    np.random.normal(0, 20000, n_samples)  # Bruit aléatoire
)

# Séparation features et target
X = df.drop('price', axis=1)
y = df['price']

# Division train/test (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entraînement du modèle Random Forest
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# Évaluation
y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: ${mae:,.2f}")
print(f"R² Score: {r2:.4f}")

# Sauvegarde
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('feature_names.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)

print("\nModèle sauvegardé avec succès!")