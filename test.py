from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the dataset
categories = ['comp.graphics', 'sci.med', 'talk.politics.misc']
train_data = fetch_20newsgroups(subset='train', categories=categories)
test_data = fetch_20newsgroups(subset='test', categories=categories)

print(type(train_data.data))
print(type(train_data.data[0]))
print(train_data.data[0])

print(type(train_data.target))
print(type(train_data.target[0]))
print(train_data.target[0])


# Extract features using TF-IDF
tfidf = TfidfVectorizer()
X_train = tfidf.fit_transform(train_data.data)
y_train = train_data.target
X_test = tfidf.transform(test_data.data)
y_test = test_data.target

# Train a KNN classifier
k = 5
clf = KNeighborsClassifier(n_neighbors=k)
clf.fit(X_train, y_train)

# Predict on the testing set and evaluate the model
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
