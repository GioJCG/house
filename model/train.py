from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from joblib import dump
import pandas as pd
import pathlib

df = pd.read_csv(pathlib.Path('data/housing.csv'))

y = df.pop('median_house_value')
X = df

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

pathlib.Path('model').mkdir(parents=True, exist_ok=True)

print('Entrenando modelo...')
clf = RandomForestRegressor(n_estimators=100, random_state=0) 
clf.fit(X_train, y_train)

dump(clf, pathlib.Path('model/housing-v1.joblib'))
print("Modelo guardado correctamente en 'model/housing-v1.joblib'")