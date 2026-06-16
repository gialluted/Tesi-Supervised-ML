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

percorso_dataset = "../../Data/Takashi2019_diabetes_type1_dataset_preprocessed.csv" + "\n"

modelli = [
    ("C++",    ["./programma"],                                              "../bin/C++",    percorso_dataset),
    ("Python", [sys.executable, "main.py"],                                  "../bin/Python", percorso_dataset),
    ("R",      ["Rscript", "progetto.R"],                                    "../bin/R",      percorso_dataset),
    ("Julia",  ["julia", "script.jl"],                                       "../bin/Julia",  percorso_dataset),
    ("Java",   ["java", "-cp", ".:../../src/Java/lib/*", "proj"],            "../bin/Java",   percorso_dataset),
]

tracker = EmissionsTracker(measure_power_secs=1, project_name="Confronto_ML")

for nome_modello, comando, cartella_lavoro, input_simulato in modelli:
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
print("Monitoraggio completato! Controlla il file 'emissions.csv' per i risultati sul consumo energetico.")
