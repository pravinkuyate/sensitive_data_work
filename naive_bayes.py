from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


import json


json_file = r"C:\Users\pravinkuyate\OneDrive\Desktop\project01\generated_data.json"


with open(json_file, 'r') as f:
    data = json.load(f)


X = [item['text'] for item in data]
y = [item['class'] for item in data]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)


classifier = MultinomialNB()
classifier.fit(X_train_vectorized, y_train)


predictions = classifier.predict(X_test_vectorized)


accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy:.2f}')


print('Classification Report:')
print(classification_report(y_test, predictions))
