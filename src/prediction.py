import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def make_predictions(X_test, model, scaler):
    # Scale the test data
    X_test_scaled = scaler.transform(X_test)

    # Make predictions
    y_pred = (model.predict(X_test_scaled) > 0.5).astype(int)

    return y_pred

def load_trained_model(model_path, scaler_path):
    # Load the trained model and scaler
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)

    return model, scaler

def evaluate_model(y_test, y_pred):
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot the confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=['Basic', 'Luxury'], yticklabels=['Basic', 'Luxury'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

