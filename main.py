import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.metrics import mean_squared_error

data = np.genfromtxt('10_7717_peerj_5665_dataYM2018_neuroblastoma.csv', 
                     delimiter=',', skip_header=1)
data = data[~np.isnan(data).any(axis=1)]
x = data[:, :-1]
y = data[:, -1]

#np.set_printoptions(threshold=np.inf)
#print(x)
#print(y)

model = LinearRegression()

loo = LeaveOneOut()
scores = cross_val_score(model, x, y, cv=loo, 
                         scoring='neg_mean_squared_error')

mse = -np.mean(scores)
rmse = np.sqrt(mse)

print(f"Errore Quadratico Medio (MSE): {mse:.4f}")
print(f"Radice dell'Errore Quadratico Medio (RMSE): {rmse:.4f}")