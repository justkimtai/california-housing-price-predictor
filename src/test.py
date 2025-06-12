import joblib
model = joblib.load('models/trained_model.pkl')
print(model.n_features_in_)  # prints 8
