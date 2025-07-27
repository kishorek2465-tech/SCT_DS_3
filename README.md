# 🌳 SCT_DS_3: Decision Tree on Bank Marketing Dataset

## 🎯 Task Objective
Build and evaluate a Decision Tree Classifier to predict whether a customer will subscribe to a term deposit based on the bank marketing dataset.

## 📁 Dataset
- File: bank.csv
- Source: UCI Machine Learning Repository
- Description: The dataset includes details from a Portuguese bank’s direct marketing campaigns (phone calls). Each row represents a client, and the target is whether they subscribed to a term deposit (y: yes/no).

## 🛠️ Tools Used
- Python
- pandas
- scikit-learn
- matplotlib
- seaborn

## 📊 Model Details
- Model: DecisionTreeClassifier (Gini index)
- Target Variable: y
- Features: All other columns
- Train-Test Split: 70% training, 30% testing
- Evaluation Metrics: Accuracy, Precision, Recall, F1-Score
- Visualization: The decision tree is saved as a PNG image.

## 🖼️ Output
The script prints:
- Model Accuracy
- Classification Report (Precision, Recall, F1-score)
- Saves the decision tree image to: plots/decision_tree.png

## ▶️ How to Run This Project
1. Open terminal in the project folder  
2. Install dependencies:  
   pip install pandas scikit-learn matplotlib seaborn  
3. Run the script:  
   python decision_tree_bank.py

## ✅ Folder Structure
SCT_DS_3/  
├── bank.csv  
├── decision_tree_bank.py  
├── plots/  
│   └── decision_tree.png  
└── README.md

## 👨‍💻 Author
Kishore K (SkillCraft Internship Task 3)
