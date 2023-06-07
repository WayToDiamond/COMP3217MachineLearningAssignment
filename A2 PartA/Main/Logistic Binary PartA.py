import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)  # Ignore future warnings
warnings.filterwarnings("ignore", category=UserWarning)  # Ignore user warnings

# Load the training data
training_data = pd.read_csv('./data/TrainingDataBinary.csv', header=None)

# Extract features and labels
X_train = training_data.iloc[:, :128].values
y_train = training_data.iloc[:, -1].values

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

# Load the testing data
testing_data = pd.read_csv('./data/TestingDataBinary.csv',  header=None)

# Extract features from testing data
X_test = testing_data.iloc[:, :128].values

# Create logistic regression object with increased max_iter and different solver
classifier = LogisticRegression(max_iter=6000, solver='liblinear')

# Train the model
classifier.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = classifier.predict(X_test)

# Save the predicted labels and input values to a file with the same format as TestingDataBinary.csv
testing_results = pd.DataFrame(X_test)
testing_results['Label'] = y_pred
testing_results.to_csv('TestingResultsBinary.csv', index=False)

# Calculate training accuracy
y_train_pred = classifier.predict(X_train)
training_accuracy = accuracy_score(y_train, y_train_pred)
print("Training Accuracy: {:.2f}%".format(training_accuracy * 100))

# Calculate the confusion matrix
cm = confusion_matrix(y_train, y_train_pred)


# Plot the confusion matrix
plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.title('Confusion Matrix')
plt.colorbar()

# Add the count values in the plot
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.xticks(range(len(set(y_train))), labels=set(y_train))
plt.yticks(range(len(set(y_train))), labels=set(y_train))
plt.show()

# Make predictions on the validation data
y_val_pred = classifier.predict(X_val)

# Calculate validation accuracy
validation_accuracy = accuracy_score(y_val, y_val_pred)
print("Validation Accuracy: {:.2f}%".format(validation_accuracy * 100))

# Calculate the confusion matrix for validation data
cm_val = confusion_matrix(y_val, y_val_pred)

# Plot the confusion matrix for validation data
plt.imshow(cm_val, interpolation='nearest', cmap='Blues')
plt.title('Validation Data Confusion Matrix')
plt.colorbar()

# Add the count values in the plot
thresh = cm_val.max() / 2.0
for i in range(cm_val.shape[0]):
    for j in range(cm_val.shape[1]):
        plt.text(j, i, format(cm_val[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm_val[i, j] > thresh else "black")
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.xticks(range(len(set(y_val))), labels=set(y_val))
plt.yticks(range(len(set(y_val))), labels=set(y_val))
plt.show()

# Compute precision, recall, and F1-score
report = classification_report(y_val, y_val_pred)
print(report)

print(y_pred)