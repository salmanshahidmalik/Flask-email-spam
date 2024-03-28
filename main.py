from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('spam_classifier_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Define a route for the home page
@app.route('/')
def home():
    return render_template('home.html')

# Define a route for classifying emails
@app.route('/classify', methods=['POST'])
def classify_email():
    print("Received request for classifying email...")
    # Check if request contains JSON data
    if request.is_json:
        print("Request contains JSON data...")
        # Get the JSON data
        data = request.get_json()
        print("Received JSON data:", data)

        # Check if 'text' key is present in the JSON data
        if 'text' in data:
            print("Text found in JSON data...")
            email_text = data['text']
            print("Email text:", email_text)

            # Preprocess the text data
            email_vectorized = vectorizer.transform([email_text])

            # Use the loaded model to make predictions
            # Use the loaded model to make predictions
            prediction = model.predict(email_vectorized)[0]
            print("Prediction:", prediction)

            if prediction == "spam":
                spam_label = "Spam"
            else:
                spam_label = "Not Spam"
            print("Spam label:", spam_label)
            # Return the prediction as JSON response
            return jsonify({'classification': spam_label})
        else:
            return jsonify({'error': 'Key "text" not found in JSON data'}), 400
    else:
        return jsonify({'error': 'Request is not JSON'}), 400

if __name__ == '__main__':
    app.run(debug=True)
