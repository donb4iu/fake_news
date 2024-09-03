import seaborn as sns
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


def label_and_concatenate(real: pd, fake: pd) -> pd:
    """
    Adds labels to the real and fake datasets and concatenates them into a single DataFrame.

    Args:
        real (DataFrame): The DataFrame containing real data.
        fake (DataFrame): The DataFrame containing fake data.

    Returns:
        DataFrame: A concatenated DataFrame with labeled real and fake data.
    """
    real['label'] = 1
    fake['label'] = 0
    data = pd.concat([real, fake])
    return data


real = pd.read_csv('app/Fake News Detection using NLP techniques/data/True.csv') # Replace with actual DataFrame
fake = pd.read_csv('app/Fake News Detection using NLP techniques/data/Fake.csv')  # Replace with actual DataFrame

data = label_and_concatenate(real, fake)

plt.figure(num="News")
plt.title("Real & Fake News")
plt.xlabel('Validity')
plt.ylabel('Count')
plt.xticks([0, 1], ['Fake', 'Real'])
sns.countplot(x='label', data=data, hue='label', legend=False)
plt.show()

print(data.isnull().sum())

# data has no null values
plt.figure(figsize = (20,10), num="News")
plt.title("Categories")
data['subject'].value_counts()
sns.countplot(x = 'subject', data=data, hue='subject')
plt.show()

plt.figure(figsize = (10,10),num="News")
plt.title("Validity")
plt.xticks([0, 1], ['Fake', 'Real'])
chart = sns.countplot(x = "label", hue = "subject" , data = data , palette = 'muted')
chart.set_xticklabels(chart.get_xticklabels(),rotation=90)
plt.show()

data['text'] = data['title'] + " " + data['text']
data = data.drop(['title', 'subject', 'date'], axis=1)

import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
english_stopwords = set(stopwords.words('english'))
# Get the text data for the desired label (label == 1)
text_data = data[data['label'] == 0].text
# Join the text data into a single string
text_string = " ".join(text_data)

wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = english_stopwords, 
                min_font_size = 10).generate(text_string) 
  
# plot the word cloud for fake news data                      
plt.figure(figsize = (8, 8), facecolor = None, num = "News") 
plt.title("Fake News")
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.show() 

# Get the text data for the desired label (label == 1)
text_data = data[data['label'] == 1].text
# Join the text data into a single string
text_string = " ".join(text_data)

wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = english_stopwords, 
                min_font_size = 10).generate(text_string) 
  
# plot the WordCloud image for genuine news data                     
plt.figure(figsize = (8, 8), facecolor = None, num = "News") 
plt.title("Real News")
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.show() 

# Classification
#splitting data for training and testing
import sklearn
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(data['text'],data['label'],test_size=0.2, random_state = 1)

# apply various models and evaluate the performance.
# Multinomial NB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import sklearn.metrics as metrics                                                 
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

pipe = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB())
])

model = pipe.fit(x_train, y_train)
prediction = model.predict(x_test)

score = metrics.accuracy_score(y_test, prediction)
print("NB accuracy:   %0.3f" % (score*100))

## Seaborn Heat Map
# Compute confusion matrix
cm = confusion_matrix(y_test, prediction)
# Compute accuracy for each cell
accuracy = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
# Create a figure with a specific number or name
fig = plt.figure(num="News")
# Create an axis object - This effectively means you are creating a single Axes object that occupies the entire figure.
ax = fig.add_subplot(111)
# Create annotation strings with both count and accuracy
annotations = np.array([f'{count}\n({acc:.3f})' for count, acc in zip(cm.flatten(), accuracy.flatten())])
annotations = annotations.reshape(cm.shape)
# Plot confusion matrix using seaborn heatmap
sns.heatmap(cm, annot=annotations, fmt='', ax=ax, cmap='Blues', cbar=True, annot_kws={"size": 10})
# Set labels and title if needed
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('NB Confusion Matrix %0.3f' % (score*100))
# Show the plot
plt.show()

#SVM
from sklearn.svm import LinearSVC
pipe = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', LinearSVC())
])

model = pipe.fit(x_train, y_train)
prediction = model.predict(x_test)

score = metrics.accuracy_score(y_test, prediction)
print("SVM accuracy:   %0.3f" % (score*100))

## Seaborn Heat Map
# Compute confusion matrix
cm = confusion_matrix(y_test, prediction)
# Compute accuracy for each cell
accuracy = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
# Create a figure with a specific number or name
fig = plt.figure(num="News")
# Create an axis object - This effectively means you are creating a single Axes object that occupies the entire figure.
ax = fig.add_subplot(111)
# Create annotation strings with both count and accuracy
annotations = np.array([f'{count}\n({acc:.3f})' for count, acc in zip(cm.flatten(), accuracy.flatten())])
annotations = annotations.reshape(cm.shape)
# Plot confusion matrix using seaborn heatmap
sns.heatmap(cm, annot=annotations, fmt='', ax=ax, cmap='Blues', cbar=True, annot_kws={"size": 10})
# Set labels and title if needed
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('SVM Confusion Matrix %0.3f' % (score*100))
# Show the plot
plt.show()

#Passive Aggressive Classifier
from sklearn.linear_model import PassiveAggressiveClassifier
pipe = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf',  PassiveAggressiveClassifier())
])

model = pipe.fit(x_train, y_train)
prediction = model.predict(x_test)

score = metrics.accuracy_score(y_test, prediction)
print("PAC accuracy:   %0.3f" % (score*100))

## Seaborn Heat Map
# Compute confusion matrix
cm = confusion_matrix(y_test, prediction)
# Compute accuracy for each cell
accuracy = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
# Create a figure with a specific number or name
fig = plt.figure(num="News")
# Create an axis object - This effectively means you are creating a single Axes object that occupies the entire figure.
ax = fig.add_subplot(111)
# Create annotation strings with both count and accuracy
annotations = np.array([f'{count}\n({acc:.3f})' for count, acc in zip(cm.flatten(), accuracy.flatten())])
annotations = annotations.reshape(cm.shape)
# Plot confusion matrix using seaborn heatmap
sns.heatmap(cm, annot=annotations, fmt='', ax=ax, cmap='Blues', cbar=True, annot_kws={"size": 10})
# Set labels and title if needed
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('PAC Confusion Matrix %0.3f' % (score*100))
# Show the plot
plt.show()