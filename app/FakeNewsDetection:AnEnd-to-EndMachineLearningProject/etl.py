import pandas as pd
import string
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

# Download stopwords
# nltk.download('stopwords')
english_stopwords = set(stopwords.words('english'))

# Load the datasets
true_news = pd.read_csv('app/FakeNewsDetection:AnEnd-to-EndMachineLearningProject/data/True.csv')
fake_news = pd.read_csv('app/FakeNewsDetection:AnEnd-to-EndMachineLearningProject/data/Fake.csv')

# Add a label column
true_news['label'] = 1
fake_news['label'] = 0

# Combine the datasets
data = pd.concat([true_news, fake_news], ignore_index=True)

# Preprocess text data
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [word for word in words if word not in english_stopwords]
    return ' '.join(words)

data['clean_text'] = data['text'].apply(preprocess_text)

# Vectorize text data using TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(data['clean_text']).toarray()
y = data['label']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Exploratory Data Analysis
#sns.countplot(data['label'])
#plt.xlabel('Label')
#plt.ylabel('Count')
#plt.title('Distribution of Fake and Real News')
#plt.show()

plt.figure(num="News")
plt.title("Distribution of Fake and Real News")
plt.xlabel('Label')
plt.ylabel('Count')
plt.xticks([0, 1], ['Fake', 'Real'])
sns.countplot(x='label', data=data, hue='label', legend=False)
plt.show()

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Train Logistic Regression model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

# Predict and evaluate Logistic Regression model
y_pred_lr = lr_model.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f'Logistic Regression Accuracy: {accuracy_lr}')
print(classification_report(y_test, y_pred_lr))

# Confusion Matrix for Logistic Regression
cm_lr = confusion_matrix(y_test, y_pred_lr)
disp_lr = ConfusionMatrixDisplay(confusion_matrix=cm_lr)
disp_lr.plot()
plt.title('Logistic Regression Confusion Matrix')
plt.show()

# Train Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict and evaluate Random Forest Classifier
y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f'Random Forest Classifier Accuracy: {accuracy_rf}')
print(classification_report(y_test, y_pred_rf))

# Confusion Matrix for Random Forest Classifier
cm_rf = confusion_matrix(y_test, y_pred_rf)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf)
disp_rf.plot()
plt.title('Random Forest Classifier Confusion Matrix')
plt.show()

import joblib

# Save the trained model
joblib.dump(rf_model, 'app/FakeNewsDetection:AnEnd-to-EndMachineLearningProject/rf_model.pkl')
joblib.dump(tfidf, 'app/FakeNewsDetection:AnEnd-to-EndMachineLearningProject/tfidf_vectorizer.pkl')