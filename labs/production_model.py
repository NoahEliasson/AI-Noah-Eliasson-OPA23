import pandas as pd
import joblib

test_samples = pd.read_csv('test_samples.csv')
model = joblib.load('trained_model.pkl')

predictions = model.predict(test_samples)

probabilities = model.predict_proba(test_samples)

results = pd.DataFrame({
    'probability class 0': probabilities[:, 0],
    'probability class 1': probabilities[:, 1],
    'prediction': predictions
})

results.to_csv('predictions.csv', index=False)