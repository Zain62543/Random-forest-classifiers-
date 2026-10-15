
# Random Forest Classifier - Beginner Project
# Author: Zain Ud Din

# 1. Import required libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

print("Dataset loaded successfully!")
print("Number of samples:", X.shape[0])
print("Number of features:", X.shape[1])
print("Flower classes:", iris.target_names)

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# 4. Create the Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
)

# 5. Train the model
model.fit(X_train, y_train)
print("\nModel training completed!")

# 6. Make predictions
y_pred = model.predict(X_test)

# 7. Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel accuracy: {accuracy:.2%}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names,
    )
)

# 8. Plot the confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names,
)

plt.title("Random Forest - Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()

# 9. Plot feature importance
plt.figure(figsize=(7, 5))
plt.barh(iris.feature_names, model.feature_importances_)
plt.title("Random Forest - Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300)
plt.show()

# 10. Predict the class of a new flower
new_flower = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(new_flower)[0]

print("\nPrediction for the new flower:")
print("Predicted class:", iris.target_names[prediction])

