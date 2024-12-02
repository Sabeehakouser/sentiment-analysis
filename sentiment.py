import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# data
data = pd.read_csv('IMDB Dataset.csv')
df = pd.DataFrame(data)
print("'--describe--'")
print(df.describe())
print("'--information--'")
print(df.info())
# Feature extraction
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['review'])
# Convert sentiment labels to 1 and 0
y = df['sentiment'].replace({'positive': 1, 'negative': 0})

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print(accuracy_score(y_test, y_pred))

# Convert predicted probabilities to binary predictions
y_pred_binary = (model.predict_proba(X_test)[:, 1] >= 0.5).astype(int)

# Convert binary predictions back to "positive" and "negative"
y_pred_labels = ['positive' if i == 1 else 'negative' for i in y_pred_binary]

# Create a DataFrame to store predictions
predictions_df = pd.DataFrame({'Actual Sentiment': y_test.replace({1: 'positive', 0: 'negative'}),
                                'Predicted Sentiment': y_pred_labels})

# Save the DataFrame to a CSV file
predictions_df.to_csv('predictions.csv', index=False)
