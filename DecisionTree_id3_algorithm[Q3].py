# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a decision tree classifier using ID3 algorithm
dt_classifier = DecisionTreeClassifier(criterion='entropy', random_state=42)
dt_classifier.fit(X_train, y_train)

# Visualize the decision tree
tree_rules = export_text(dt_classifier, feature_names=iris.feature_names)
print("Decision Tree Rules:")
print(tree_rules)

# Make predictions on the test set
y_pred = dt_classifier.predict(X_test)

# Evaluate the performance
accuracy = (y_pred == y_test).mean()
print(f"\nAccuracy: {accuracy:.2f}")

# Classify a new sample
new_sample = [[5.1, 3.5, 1.4, 0.2]]  # Replace with your own values
predicted_class = dt_classifier.predict(new_sample)
print(f"\nPredicted class for the new sample: {iris.target_names[predicted_class][0]}")
