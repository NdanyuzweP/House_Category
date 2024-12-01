import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def preprocess_data(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)

    # Handle missing values by imputing with median
    imputer = SimpleImputer(strategy='median')
    data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

    # Handle categorical variables by one-hot encoding (if any)
    data = pd.get_dummies(data, drop_first=True)

    # Separate features and target variable (assuming 'Price' is the target)
    X = data.drop('Price', axis=1)
    y = data['Price']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features using StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler

