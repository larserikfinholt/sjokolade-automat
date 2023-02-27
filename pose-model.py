import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load your data into pandas DataFrames
df_left = pd.read_csv('data/RaiseLeftArm.csv' , delimiter=';',header=None)
df_right = pd.read_csv('data/RaiseRightArm.csv', delimiter=';',header=None)
df_both = pd.read_csv('data/RiseBothArms.csv', delimiter=';',header=None)
df_test = pd.read_csv('data/Testing.csv', delimiter=';',header=None)

print(df_left.head())
print(df_right.head())
print(df_both.head())

# Add a 'class' column to each DataFrame
df_left['class'] = 'RaiseLeftArm'
df_right['class'] = 'RaiseRightArm'
df_both['class'] = 'RaiseBothArms'

# Concatenate the DataFrames into one
data = pd.concat([df_left, df_right, df_both])
print(data.head())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop('class', axis=1), data['class'], test_size=0.3, random_state=42)

# Create a decision tree classifier
classifier = DecisionTreeClassifier(random_state=42)

# Train the classifier on the training data
classifier.fit(X_train, y_train)

# Save the classifier model to a binary file
with open('classifier.pickle', 'wb') as f:
    pickle.dump(classifier, f)

# Use the classifier to make predictions on the test data
y_pred = classifier.predict(X_test)

# Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

test_pred = classifier.predict(df_test)

print("result",test_pred)
