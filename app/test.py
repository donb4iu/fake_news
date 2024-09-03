import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Sample data
y_true = np.array([1, 0, 1, 0, 1, 1, 0, 0, 1, 0])
y_pred = np.array([1, 0, 0, 0, 1, 1, 0, 1, 1, 0])

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Create a figure with a specific number or name
fig = plt.figure(num="News")

# Create an axis object
ax = fig.add_subplot(111)

# Plot confusion matrix using seaborn heatmap
sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues')

# Set labels and title if needed
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')

# Show the plot
plt.show()

