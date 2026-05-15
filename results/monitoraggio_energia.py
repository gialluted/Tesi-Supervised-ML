import subprocess
import sys

libraries = ["codecarbon"]

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

from codecarbon import EmissionsTracker

percorso_dataset = r"..\..\data\Takashi2019_diabetes_type1_dataset_preprocessed.csv" + "\n"

modelli = {
    "C++_Model":    ([r"..\bin\C++\programma.exe"], r"..\bin\C++", percorso_dataset),
    "Python_Model": ([sys.executable, "main.py"], r"..\bin\Python", percorso_dataset),
    "R_Model":      (["Rscript", "progetto.R"], r"..\bin\R", percorso_dataset),
    "Julia_Model":  (["Julia", "script.jl"], r"..\bin\Julia", percorso_dataset),
    "Java_Model":   (["java", "-cp", r".;..\..\src\Java\lib\*", "proj"], r"..\bin\Java", percorso_dataset)
}

tracker = EmissionsTracker(measure_power_secs=1, project_name="Confronto_ML")
    
for nome_modello, (comando, cartella_lavoro, input_simulato) in modelli.items():
    print(f"{'-'*50}")
    print(f"Avvio monitoraggio per: {nome_modello}")
    print(f"Cartella di lavoro: {cartella_lavoro}")
        
    tracker.start_task(nome_modello)
        
    try:
        subprocess.run(
            comando, 
            cwd=cartella_lavoro,
            input=input_simulato,
            text=True,
            check=True
        )
        print(f"Esecuzione di {nome_modello} completata con successo.")
            
    except subprocess.CalledProcessError as e:
        print(f"ERRORE: Il modello {nome_modello} ha generato un errore durante l'esecuzione: {e}")
    except FileNotFoundError:
        print(f"ERRORE: Impossibile trovare il comando o la cartella per {nome_modello}.")
    finally:
        tracker.stop_task()
            
tracker.stop()
print(f"{'-'*50}")
print("Monitoraggio completato! Controlla il file 'emissions.csv' per i risultati sul consumo energetico.")