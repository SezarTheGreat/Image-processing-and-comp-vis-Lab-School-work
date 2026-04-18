from sklearn.neighbors import KNeighborsClassifier

# Step 2: Create simple dataset
# X = features (2D points), y = class labels
X = [[1, 2], [2, 3], [3, 4], [6, 7], [7, 8], [8, 9]]
y = [0, 0, 0, 1, 1, 1]

# Step 3: Create KNN model
# n_neighbors = 3 means we check 3 nearest points
knn = KNeighborsClassifier(n_neighbors=3)

# Step 4: Train the model
knn.fit(X, y)

x1 = int(input("Enter first value: "))
x2 = int(input("Enter second value: "))

# Step 5: Predict
prediction = knn.predict([[x1, x2]])

# Step 6: Print result
print("Predicted class:", prediction)