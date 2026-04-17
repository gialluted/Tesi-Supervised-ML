import subprocess
import sys

libraries = ["numpy", "scikit-learn"]

for lib in libraries:
    try:
        __import__(lib)
        print(f"{lib} è già installato")
    except ImportError:
        print(f"{lib} non trovato. Installazione in corso...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
            print(f"{lib} installato con successo")
        except subprocess.CalledProcessError as e:
            print(f"Errore nell'installazione di {lib}: {e}")
            sys.exit(1)

import time
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score, matthews_corrcoef

for i in range(2):
    start = time.time()

    data = np.genfromtxt('C:\\Users\\giall\\Documents\\GitHub\\Tesi-Supervised-ML\\data\\Takashi2019_diabetes_type1_dataset_preprocessed.csv', 
                     delimiter=',', skip_header=1)

    variabili = data[:, :-1]
    outcome = data[:, -1]

    #np.set_printoptions(threshold=np.inf)
    #print(variabili)
    #print(outcome)

    for indice_colonna in range(variabili.shape[1]):
        colonna = variabili[:, indice_colonna]
    
        # Identifica i valori non mancanti
        valori_validi = colonna[~np.isnan(colonna)]
    
        if len(valori_validi) == 0:
            continue
    
        # Conta i valori mancanti
        numero_mancanti = np.isnan(colonna).sum()
    
        if numero_mancanti == 0:
            continue
    
        # Verifica se la colonna è binaria (contiene solo 0 e 1)
        valori_unici = np.unique(valori_validi)
        e_binaria = np.all(np.isin(valori_unici, [0, 1]))
    
        if e_binaria:
            # Colonna binaria: usa la MEDIANA
            valore_per_imputazione = np.median(valori_validi)
            tipo_imputazione = "mediana"
        else:
            # Colonna reale: usa la MEDIA
            valore_per_imputazione = np.mean(valori_validi)
            tipo_imputazione = "media"
    
        # Sostituisci i valori mancanti
        variabili[np.isnan(colonna), indice_colonna] = valore_per_imputazione

    predictions = cross_val_predict(LinearRegression(), variabili, outcome, cv=LeaveOneOut())

    for i in range(len(predictions)):

        if predictions[i] > 0.5:

            predictions[i] = 1

        else:

            predictions[i] = 0

    #print(predictions)

    mcc = matthews_corrcoef(outcome, predictions)
    print(f"Coefficiente di Correlazione di Matthews (MCC): {mcc}")

    print("Durata dell'esecuzione del programma: %s secondi" % (time.time() - start))