import subprocess
from codecarbon import EmissionsTracker

def esegui_e_monitora():
    # 1. Definisci il percorso del dataset da passare in automatico a tutti gli script.
    # Il "\n" alla fine è fondamentale: simula la pressione del tasto "Invio"
    percorso_dataset = r"C:\Users\giall\Documents\GitHub\Tesi-Supervised-ML\data\Takashi2019_diabetes_type1_dataset_preprocessed.csv" + "\n"

    # 2. Configura i modelli.
    # Struttura: "Nome Modello": (["comando", "nome_file"], r"Cartella_di_lavoro", input_automatico)
    # ATTENZIONE: Modifica i percorsi delle cartelle (cwd) con quelli reali del tuo PC.
    modelli = {
        "C++_Model":    (["programma.exe"], r"..\bin\C++", percorso_dataset) # Su Mac/Linux usa ["./programma.exe"]
    }

    # 3. Inizializza il tracker di CodeCarbon (campionamento ogni 1 secondo)
    tracker = EmissionsTracker(measure_power_secs=1, project_name="Confronto_ML")

    # 4. Esegui i modelli uno alla volta
    for nome_modello, (comando, cartella_lavoro, input_simulato) in modelli.items():
        print(f"{'-'*50}")
        print(f"Avvio monitoraggio per: {nome_modello}")
        print(f"Cartella di lavoro: {cartella_lavoro}")
        
        # Inizia a misurare i consumi specifici per questo modello
        tracker.start_task(nome_modello)
        
        try:
            # Avvia il processo in background
            subprocess.run(
                comando, 
                cwd=cartella_lavoro,   # Si "sposta" nella cartella dello script
                input=input_simulato,  # Scrive in automatico il percorso del dataset
                text=True,             # Tratta i dati di input/output come testo leggibile
                check=True             # Genera un errore se lo script ML va in crash
            )
            print(f"Esecuzione di {nome_modello} completata con successo.")
            
        except subprocess.CalledProcessError as e:
            print(f"ERRORE: Il modello {nome_modello} ha generato un errore durante l'esecuzione: {e}")
        except FileNotFoundError:
            print(f"ERRORE: Impossibile trovare il comando o la cartella per {nome_modello}.")
        finally:
            # Ferma la misurazione per questo modello (avviene anche in caso di errore)
            tracker.stop_task()
            
    # 5. Ferma il monitoraggio globale e salva i dati
    tracker.stop()
    print(f"{'-'*50}")
    print("Monitoraggio completato! Controlla il file 'emissions.csv' per i risultati sul consumo energetico.")

if __name__ == "__main__":
    esegui_e_monitora()