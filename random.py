from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

# 1. Training data
X_train = [
    "Win money now",
    "Claim your free prize",
    "Congratulations you won a cash prize",
    "Click here to claim your reward",
    "Meeting is scheduled for tomorrow",
    "Please find the project report",
    "Team meeting is scheduled for Monday",
    "Please review the attached document"
]

# 2. Correct labels
y_train = [
    "Spam",
    "Spam",
    "Spam",
    "Spam",
    "Not Spam",
    "Not Spam",
    "Not Spam",
    "Not Spam"
]

# 3. Convert text into numerical features
vectorizer = CountVectorizer()
X_numerical = vectorizer.fit_transform(X_train)

# 4. Create the Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# 5. Train the model
model.fit(X_numerical, y_train)

# 6. Test a new email
new_message = ["Claim your free money now"]

new_message_numerical = vectorizer.transform(new_message)

# 7. Predict the email category
prediction = model.predict(new_message_numerical)

# 8. Display the result
print("Email:", new_message[0])
print("Predicted label:", prediction[0])
