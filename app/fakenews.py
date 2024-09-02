import seaborn as sns
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


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


real = pd.read_csv('app/data/True.csv') # Replace with actual DataFrame
fake = pd.read_csv('app/data/Fake.csv')  # Replace with actual DataFrame

data = label_and_concatenate(real, fake)

plt.figure(num="News")
plt.title("Real & Fake News")
plt.xlabel('Validity')
plt.ylabel('Count')
plt.xticks([0, 1], ['Fake', 'Real'])
sns.countplot(x='label', data=data, hue='label')
plt.show()

print(data.isnull().sum())

# data has no null values
plt.figure(figsize = (20,10), num="News")
plt.title("Categories")
data['subject'].value_counts()
sns.countplot(x = 'subject', data=data, hue='subject')
plt.show()
