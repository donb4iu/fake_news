from flask import Flask, request, jsonify
import joblib
import nltk
from nltk.corpus import stopwords
import string
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

app = Flask(__name__)

# Load the trained model and TF-IDF vectorizer
model = joblib.load('rf_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')
#nltk.download('stopwords')
english_stopwords = set(stopwords.words('english'))

# Preprocess text data
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
#    words = [word for word in words if word not in stopwords.words('english')]
    words = [word for word in words if word not in english_stopwords]
    return ' '.join(words)

def scrape_text(url):
    try:
        with tqdm(total=3, desc="Scraping Progress", unit="step") as pbar:
            response = requests.get(url)
            pbar.update(1)  # Update the progress after getting the response
            soup = BeautifulSoup(response.text, 'html.parser')
            pbar.update(1)  # Update the progress after parsing the HTML
            # Remove script, style, and image tags to clean the text
            for element in soup(['script', 'style', 'img']):
                element.extract()
            pbar.update(1)  # Update the progress after cleaning the text
            # Get the text and strip leading/trailing whitespace
            clean_text = soup.get_text(separator=' ').strip()
            json_object = {"text": clean_text}
            return json_object
    except requests.RequestException as e:
        print(f"An error occurred: {e}")
        return None

# {'text':' text to predict'}
@app.route('/predict/json', methods=['POST'])
def predict_json():
    data = request.get_json(force=True)
    if 'text' not in data:
        return jsonify({"error": "Missing 'text' field in JSON input."}), 400
    else:
        data['clean_text'] = preprocess_text(data['text'])
        print(data['clean_text'])
    vectorized_data = tfidf.transform([data['clean_text']]).toarray()
    prediction = model.predict(vectorized_data)
    return jsonify(prediction=int(prediction[0]))

# {'url':' url to website content to predict'}
@app.route('/predict/url', methods=['POST'])
def predict_url():
    data = request.get_json(force=True)
    if 'url' not in data:
        return jsonify({"error": "Missing 'url' field in JSON input."}), 400
    else:
        json_object = scrape_text(data['url'])
        data['clean_text'] = preprocess_text(json_object['text'])
        print(data['clean_text'])
    vectorized_data = tfidf.transform([data['clean_text']]).toarray()
    prediction = model.predict(vectorized_data)
    return jsonify(prediction=int(prediction[0]))


if __name__ == '__main__':
    app.run(debug=True)