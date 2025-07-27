import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("bank.csv", sep=";")

# Encode categorical columns (except target)
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    if column != 'y':
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

# Encode target column (yes/no → 1/0)
df['y'] = df['y'].map({'yes': 1, 'no': 0})

# Split data into features and target
X = df.drop('y', axis=1)
y = df['y']

# Train-test split (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42, max_depth=5)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Plot and save the decision tree
plt.figure(figsize=(20, 10))
plot_tree(clf, 
          feature_names=X.columns, 
          class_names=['No', 'Yes'], 
          filled=True,
          rounded=True)

# Save as PNG
plt.savefig("decision_tree.png", dpi=300, bbox_inches='tight')
plt.show()
