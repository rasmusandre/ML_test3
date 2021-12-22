from sklearn.ensemble import RandomForestRegressor
from joblib import dump, load

rf = RandomForestRegressor()

dump(rf, 'rf_model.joblib')

df_loaded = load('rf_model.joblib')
