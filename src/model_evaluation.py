def evaluate_best_model(best_model, X_test, y_test):
    predictions = best_model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    print(f"Test RMSE: {rmse}")