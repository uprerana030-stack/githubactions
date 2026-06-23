import  os
from dotenv import load_dotenv
import pandas as pd
import joblib
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(env_path)
def fraud_detection_train(file_path):
    """
    Function to train a fraud detection model using the provided dataset.

    Args:
        file_path (str): The path to the CSV file containing the training data.

    Returns:
        model: The trained fraud detection model.
    """
    # Load the dataset
    data = pd.read_csv(file_path)

    # Preprocess the data (this is a placeholder, actual preprocessing steps will depend on the dataset)
    # For example, you might want to handle missing values, encode categorical variables, etc.
    
    # 1. Separate your features and your target label
    X = data.drop('label', axis=1)
    y = data['label']

    # 2. Drop the transaction_id column entirely
    X = X.drop('transaction_id', axis=1)

    # 3. One-hot encode categorical features ('merchant_category' and 'location')
    X = pd.get_dummies(X, columns=['merchant_category', 'location'], drop_first=True)

    # Split the dataset into training and testing sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train a machine learning model (e.g., decision tree)
    #decision tree model
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train) 
    #prediction
    y_pred = model.predict(X_test)
    print("Model trained successfully.")
    #accuracy,f1_score,recall_score,precision_score
    from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
    print(f"Recall: {recall}")
    print(f"Precision: {precision}")    
    #confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix: \n{cm}")
    os.makedirs("outputs", exist_ok=True)

    model_path = os.getenv('model_path')

    artifact = {
            "model": model,
            "feature_columns": X.columns.tolist(),
            "metrics": {
                "accuracy": accuracy,
                "f1_score": f1,
                "precision": precision,
                "recall": recall
            }
        }



    joblib.dump(artifact, model_path)

    print(f"\nModel saved to: {model_path}")
    return model
if __name__ == "__main__":
    # Example usage
    file_path = file_path = "src/azuremlapp/resources/fraud_transactions.csv"  # Ensure this environment variable is set in your .env file
    if file_path:
        model = fraud_detection_train(file_path)
    else:
        print("Please set the TRAINING_DATA_PATH environment variable in the .env file.")
