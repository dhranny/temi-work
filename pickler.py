import joblib
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

model = joblib.load("scam_detection_model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

def preprocess_input(text):
    # You can add any necessary preprocessing steps here
    return text

def predict(text):
    # Preprocess the input text
    preprocessed_text = preprocess_input(text)

    # Check if the vectorizer is fitted
    try:
        check_is_fitted(vectorizer, 'transform')
    except NotFittedError:
        print("Error: The TF-IDF vectorizer is not fitted. Please fit it before using.")
        return None

    # Transform the preprocessed text using the vectorizer
    vectorized_text = vectorizer.transform([preprocessed_text])

    # Make a prediction using the model
    prediction = model.predict(vectorized_text)

    return prediction[0]

if __name__ == "__main__":
    # Example usage
    user_input = input("Enter the text for prediction: ")
    
    result = predict(user_input)
    if result is not None:
        print(f"Prediction: {result}")
