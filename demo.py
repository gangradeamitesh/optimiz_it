import numpy as np

from sklearn.model_selection import train_test_split
from boston_dataset import BostonHousingDataset
import pandas as pd
from optimiz.linear_model import LinearModel
from optimiz.losses import mse_loss
import cProfile
import pstats
import io
import matplotlib.pyplot as plt

boston_housing = BostonHousingDataset()
boston_dataset = boston_housing.load_dataset()
boston_dataset.keys(), boston_dataset['DESCR']
# Load the Boston Housing Dataset from sklearn
# Create the dataset
boston = pd.DataFrame(boston_dataset['data'], columns=boston_dataset['feature_names'])
boston['MEDV'] = boston_dataset['target']
boston.head()

from sklearn.model_selection import train_test_split
X = boston.to_numpy()
X = np.delete(X, 13, 1)
y = boston['MEDV'].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=5)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
pr = cProfile.Profile()
pr.enable()

model = LinearModel(learning_rate=0.000001 , iterations=3000 , tolerance=0.0000001 , optimizer_type="gradient_descent" )


loss_history = model.fit(X_train , y_train , scale=True)

y_pred = model.predict(X_test)

# Print the learned parameters
print(f"Intercept: {model.weights[0]}")
print(f"Coefficient: {model.weights[1]}")
print(mse_loss(y_test , y_pred))
pr.disable()

s = io.StringIO()
sortby = pstats.SortKey.CUMULATIVE
ps = pstats.Stats(pr , stream=s).sort_stats(sortby)
ps.print_stats()
print(s.getvalue())

# plt.scatter(X_test, y_test, color='blue', label='Actual')
# plt.plot(X_test, y_pred, color='red', label='Predicted')
# plt.legend()
# plt.xlabel('X')
# plt.ylabel('y')
# plt.title('Linear Regression Results')
# plt.show()

# plt.figure(figsize=(15,5))
# plt.plot(list(range(1,3001)) , loss_history , label = 'Loss' , color = 'blue')
# plt.title('Loss History')
# plt.xlabel('Iterations')
# plt.ylabel('Loss')
# plt.grid()
# plt.legend()
# plt.show()