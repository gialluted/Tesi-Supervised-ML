start = time.time()

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score, matthews_corrcoef
import time

data = np.genfromtxt('data/10_7717_peerj_5665_dataYM2018_neuroblastoma.csv', 
                     delimiter=',')
data = data[~np.isnan(data).any(axis=1)]
variabili = data[:, :-1]
outcome = data[:, -1]

#np.set_printoptions(threshold=np.inf)
#print(variabili)
#print(outcome)

predictions = cross_val_predict(LinearRegression(), variabili, outcome, cv=LeaveOneOut())

#r2 = r2_score(outcome, predictions)
#mse = mean_squared_error(outcome, predictions)
#rmse = np.sqrt(mse)
#print(f"R-squared (R2) medio: {r2}")
#print(f"Errore Quadratico Medio (MSE): {mse}")
#print(f"Radice dell'Errore Quadratico Medio (RMSE): {rmse}")

for i in range(len(predictions)):

    if predictions[i] > 0.5:

        predictions[i] = 1

    else:

        predictions[i] = 0

#print(predictions)

mcc = matthews_corrcoef(outcome, predictions)
print(f"Coefficiente di Correlazione di Matthews (MCC): {mcc}")

print("Durata dell'esecuzione del programma: %s secondi" % (time.time() - start))