import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

# Load the training data
training_data = pd.read_csv('./data/TrainingDataMulti.csv', header=None)

# Extract features and labels
X = training_data.iloc[:, :128].values
y = training_data.iloc[:, -1].values

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# Load the testing data
testing_data = pd.read_csv('./data/TestingDataMulti.csv', header=None)

# Extract features from testing data
X_test = testing_data.iloc[:, :128].values

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Create random forest classifier with (default) 100 trees
classifier = RandomForestClassifier(n_estimators=271, random_state=0)

# Train the model
classifier.fit(X_train_scaled, y_train)

# Make predictions on the training data
y_train_pred = classifier.predict(X_train_scaled)

# Calculate training accuracy
training_accuracy = accuracy_score(y_train, y_train_pred)
print("Training Accuracy: {:.2f}%".format(training_accuracy * 100))

# Create a confusion matrix for training data
cm_train = confusion_matrix(y_train, y_train_pred)

# Plot the confusion matrix for training data
plt.imshow(cm_train, interpolation='nearest', cmap='Blues')
plt.title('Training Data Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')

# Add text annotations to the confusion matrix plot
thresh = cm_train.max() / 2
for i in range(cm_train.shape[0]):
    for j in range(cm_train.shape[1]):
        plt.text(j, i, format(cm_train[i, j], 'd'), ha="center", va="center",
                 color="white" if cm_train[i, j] > thresh else "black")

plt.xticks([0, 1, 2])
plt.yticks([0, 1, 2])
plt.show()

# Make predictions on the validation data
y_val_pred = classifier.predict(X_val_scaled)

# Calculate validation accuracy
validation_accuracy = accuracy_score(y_val, y_val_pred)
print("Validation Accuracy: {:.2f}%".format(validation_accuracy * 100))

# Create a confusion matrix for validation data
cm_val = confusion_matrix(y_val, y_val_pred)

# Plot the confusion matrix for validation data
plt.imshow(cm_val, interpolation='nearest', cmap='Blues')
plt.title('Validation Data Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')

# Add text annotations to the confusion matrix plot
thresh = cm_val.max() / 2
for i in range(cm_val.shape[0]):
    for j in range(cm_val.shape[1]):
        plt.text(j, i, format(cm_val[i, j], 'd'), ha="center", va="center",
                 color="white" if cm_val[i, j] > thresh else "black")

plt.xticks([0, 1, 2])
plt.yticks([0, 1, 2])
plt.show()

# Make predictions on the testing data
y_pred = classifier.predict(X_test_scaled)

# Save the predicted labels to a file
testing_results = pd.DataFrame(np.hstack((X_test_scaled, y_pred.reshape(-1, 1))))
testing_results.to_csv('TestingResultsMulti.csv', index=False)

# Show the best n_estimators for rfc
# cross = []
# for i  in range(0,300,10):
#     rfc = RandomForestClassifier(n_estimators=i+1, random_state=0)
#     cross_score = cross_val_score(rfc, X_train_scaled, y_train, cv=5).mean()
#     cross.append(cross_score)
#
# plt.plot(range(1,301,10),cross)
# plt.xlabel('n_estimators')
# plt.ylabel('acc')
# plt.show()
# print((cross.index(max(cross))*10)+1,max(cross))

# Compute precision, recall, and F1-score
report = classification_report(y_val, y_val_pred)
print(report)

#print results
print(y_pred)
