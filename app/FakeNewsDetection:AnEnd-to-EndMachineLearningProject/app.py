from flask import Flask, request, jsonify
import joblib
import nltk
from nltk.corpus import stopwords
import string

app = Flask(__name__)

# Load the trained model and TF-IDF vectorizer
model = joblib.load('rf_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')
nltk.download('stopwords')

# Preprocess text data
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    if 'text' not in data:
        return jsonify({"error": "Missing 'text' field in JSON input."}), 400
    else:
        data['clean_text'] = preprocess_text(data['text'])
        print(data['clean_text'])
    vectorized_data = tfidf.transform([data['clean_text']]).toarray()
    prediction = model.predict(vectorized_data)
    return jsonify(prediction=int(prediction[0]))

if __name__ == '__main__':
    app.run(debug=True)