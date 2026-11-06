
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Load dataset
data = pd.read_csv("student.csv")

print("Student Dataset:")
print(data)

# 2. Separate input and output
X = data[["Study_Hours", "Attendance"]]
y = data["Result"]

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Feature scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Apply KNN algorithm
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# 6. Make predictions
y_pred = model.predict(X_test)

# 7. Check accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy * 100, "%")

# 8. Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 9. Predict new student
new_student = pd.DataFrame({
    "Study_Hours": [7],
    "Attendance": [80]
})

new_student_scaled = scaler.transform(new_student)

prediction = model.predict(new_student_scaled)

print("\nNew Student Prediction:", prediction[0])
