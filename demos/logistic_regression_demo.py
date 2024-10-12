from sklearn.datasets import load_iris , load_breast_cancer
from sklearn.model_selection import train_test_split
from optimiz_it.optimiz.linear_model import LinearModel

data = load_breast_cancer()
X = data.data
y = data.target
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=42)

print(X)

model = LinearModel(iterations=10 , tolerance=0.00001 , optimizer_type="newton")

model.fit(X_train , y_train , scale=True)
