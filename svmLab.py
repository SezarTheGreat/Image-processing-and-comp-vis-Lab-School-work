from sklearn.svm import SVC

# Dataset
X = [[1, 2], [2, 3], [3, 4], [6, 7], [7, 8], [8, 9]]
y = [0, 0, 0, 1, 1, 1]

# Train
model = SVC(kernel="linear", probability=True)
model.fit(X, y)

# User input
x1 = int(input("Enter first value: "))
x2 = int(input("Enter second value: "))

# Predict
prediction = model.predict([[x1, x2]])
probability = model.predict_proba([[x1, x2]])

print("Predicted class:", prediction[0])
print("Probabilities:", probability[0])