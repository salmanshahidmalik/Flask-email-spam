import requests

# URL of your Flask application
url = 'http://127.0.0.1:5000/classify'

# Email text to classify
email_text = "Your email text goes here..."

# Send POST request to classify email
response = requests.post(url, json={'text': email_text})

# Print the response
print(response.json())
