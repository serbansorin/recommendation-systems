import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load the data
data = pd.read_csv("customers.csv")

# Data preprocessing
data = pd.get_dummies(data, columns=["job", "marital", "education", "housing", "loan"])
data["y"] = data["y"].map({"no": 0, "yes": 1})

# Split the data into features and target variable
X = data.iloc[:, :-1]
y = data["y"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model training
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Joe's information
joe_data = pd.DataFrame(
    {
        "job": ["management"],
        "marital": ["married"],
        "education": ["tertiary"],
        "housing": ["yes"],
        "loan": ["no"],
        "duration": [200],  # Example duration value
    }
)

# Encode Joe's data
joe_data = pd.get_dummies(
    joe_data, columns=["job", "marital", "education", "housing", "loan"]
)

# Align Joe's data with the training data columns
joe_data = joe_data.reindex(columns=X.columns, fill_value=0)

# Make predictions for Joe
joe_pred = model.predict(joe_data)
if joe_pred[0] == 1:
    print("Joe is predicted to respond 'yes'.")
else:
    print("Joe is predicted to respond 'no'.")
