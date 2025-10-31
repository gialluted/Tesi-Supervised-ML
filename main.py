import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import matthews_corrcoef

data = np.genfromtxt('10_7717_peerj_5665_dataYM2018_neuroblastoma.csv', 
                     delimiter=',')
data = data[~np.isnan(data).any(axis=1)]
input = data[:, :-1]
output = data[:, -1]

#np.set_printoptions(threshold=np.inf)
#print(input)
#print(output)

predictions = cross_val_predict(LinearRegression(), input, output, cv=LeaveOneOut())
#r2 = r2_score(output, predictions)
#mse = mean_squared_error(output, predictions)
#rmse = np.sqrt(mse)
print(predictions)

#mcc = matthews_corrcoef(output, predictions)
#print(f"MCC: {mcc}")

#print(f"R-squared (R2) medio: {r2}")
#print(f"Errore Quadratico Medio (MSE): {mse}")
#print(f"Radice dell'Errore Quadratico Medio (RMSE): {rmse}")