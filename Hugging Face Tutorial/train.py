import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

iris = load_iris()
x, y = iris.data, iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

acc = model.score(x_test, y_test)
print(f"Model Accuracy: {acc:.2%}")

joblib.dump(model, "model.pkl")
joblib.dump(list(iris.target_names), "labels.pkl")


print("Saved model.pkl and labels.pkl")