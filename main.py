import numpy as np
from sklearn.linear_model import LinearRegression

data = np.genfromtxt('10_7717_peerj_5665_dataYM2018_neuroblastoma.csv', 
                     delimiter=',', skip_header=1)
data = data[~np.isnan(data).any(axis=1)]
x = data[:, :-1]
y = data[:, 1]

#np.set_printoptions(threshold=np.inf)
#print(x)
#print(y)

model = LinearRegression().fit(x, y)

#r_sq = model.score(x, y)
#print(f"Coefficiente di determinazione: {r_sq}")
#print(f"Intercetta: {model.intercept_}")
#print(f"Coefficienti: {model.coef_}")