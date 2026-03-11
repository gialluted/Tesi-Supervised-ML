import weka.classifiers.functions.LinearRegression;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;

/**
 * Analisi predittiva su dati di neuroblastoma usando Regressione Lineare
 * con validazione Leave-One-Out e calcolo del coefficiente MCC
 * 
 * Con imputazione dei valori mancanti:
 * - Colonne binarie (0/1): imputazione con mediana
 * - Colonne reali: imputazione con media
 * - Outcome: rimuove righe con valori mancanti
 */
public class proj {

    /**
     * Verifica la presenza delle dipendenze necessarie
     */
    public static void verificaDipendenze() {
        System.out.println("=== Verifica dipendenze ===\n");
        
        try {
            Class.forName("weka.classifiers.functions.LinearRegression");
            System.out.println("✓ Weka è disponibile");
        } catch (ClassNotFoundException e) {
            System.err.println("✗ Weka non trovata!");
            System.err.println("\nERRORE: La libreria Weka è necessaria.");
            System.err.println("Consulta i commenti del file per istruzioni.\n");
            System.exit(1);
        }
        
        System.out.println("\n" + "=".repeat(60) + "\n");
    }

    /**
     * Verifica se una colonna contiene solo valori binari (0 e 1)
     */
    public static boolean eColonnaBinaria(Instances dati, int indiceAttributo) {
        for (int i = 0; i < dati.numInstances(); i++) {
            if (!dati.instance(i).isMissing(indiceAttributo)) {
                double valore = dati.instance(i).value(indiceAttributo);
                if (valore != 0.0 && valore != 1.0) {
                    return false;
                }
            }
        }
        return true;
    }

    /**
     * Calcola la media dei valori non mancanti
     */
    public static double calcolaMedia(Instances dati, int indiceAttributo) {
        double somma = 0.0;
        int conteggio = 0;
        
        for (int i = 0; i < dati.numInstances(); i++) {
            if (!dati.instance(i).isMissing(indiceAttributo)) {
                somma += dati.instance(i).value(indiceAttributo);
                conteggio++;
            }
        }
        
        return conteggio > 0 ? somma / conteggio : 0.0;
    }

    /**
     * Calcola la mediana dei valori non mancanti
     */
    public static double calcolaMediana(Instances dati, int indiceAttributo) {
        ArrayList<Double> valori = new ArrayList<>();
        
        for (int i = 0; i < dati.numInstances(); i++) {
            if (!dati.instance(i).isMissing(indiceAttributo)) {
                valori.add(dati.instance(i).value(indiceAttributo));
            }
        }
        
        if (valori.isEmpty()) return 0.0;
        
        valori.sort(Double::compare);
        int dimensione = valori.size();
        
        if (dimensione % 2 == 0) {
            return (valori.get(dimensione/2 - 1) + valori.get(dimensione/2)) / 2.0;
        } else {
            return valori.get(dimensione/2);
        }
    }

    /**
     * Rimuove le righe con outcome mancante
     */
    public static void rimuoviRigheConOutcomeMancante(Instances dati) {
        int indiceClasse = dati.classIndex();
        int righeRimosse = 0;
        
        // Rimuovi dalla fine per non alterare gli indici
        for (int i = dati.numInstances() - 1; i >= 0; i--) {
            if (dati.instance(i).isMissing(indiceClasse)) {
                dati.delete(i);
                righeRimosse++;
            }
        }
        
        if (righeRimosse > 0) {
            System.out.printf("⚠ Rimozione di %d righe con outcome mancante...%n", righeRimosse);
            System.out.printf("✓ Righe rimanenti: %d%n", dati.numInstances());
        }
    }

    /**
     * Imputa i valori mancanti nelle caratteristiche
     */
    public static void imputaValoriMancanti(Instances dati) {
        System.out.println("\nImputazione dei valori mancanti nelle caratteristiche...");
        
        int numeroAttributi = dati.numAttributes();
        
        // Per ogni attributo (esclusa la classe/outcome)
        for (int indiceAttr = 0; indiceAttr < numeroAttributi - 1; indiceAttr++) {
            int numeroMancanti = 0;
            
            // Conta i valori mancanti
            for (int i = 0; i < dati.numInstances(); i++) {
                if (dati.instance(i).isMissing(indiceAttr)) {
                    numeroMancanti++;
                }
            }
            
            if (numeroMancanti == 0) continue;
            
            // Determina se la colonna è binaria
            boolean eBinaria = eColonnaBinaria(dati, indiceAttr);
            double valorePerImputazione;
            String tipoImputazione;
            
            if (eBinaria) {
                // Colonna binaria: usa MEDIANA
                valorePerImputazione = calcolaMediana(dati, indiceAttr);
                tipoImputazione = "mediana";
            } else {
                // Colonna reale: usa MEDIA
                valorePerImputazione = calcolaMedia(dati, indiceAttr);
                tipoImputazione = "media";
            }
            
            // Sostituisci i valori mancanti
            for (int i = 0; i < dati.numInstances(); i++) {
                if (dati.instance(i).isMissing(indiceAttr)) {
                    dati.instance(i).setValue(indiceAttr, valorePerImputazione);
                }
            }
            
            System.out.printf("  Attributo %d (%s): %d valori → %s = %.4f%n",
                            indiceAttr, dati.attribute(indiceAttr).name(), 
                            numeroMancanti, tipoImputazione, valorePerImputazione);
        }
        
        System.out.println("\n✓ Imputazione completata");
    }

    /**
     * Carica dati dal CSV
     */
    public static Instances caricaDatiDaCSV(String percorsoFile) throws Exception {
        File file = new File(percorsoFile);
        if (!file.exists()) {
            throw new FileNotFoundException("File non trovato: " + percorsoFile);
        }

        CSVLoader caricatore = new CSVLoader();
        caricatore.setSource(file);
        Instances dati = caricatore.getDataSet();

        if (dati.classIndex() == -1) {
            dati.setClassIndex(dati.numAttributes() - 1);
        }

        return dati;
    }

    /**
     * Calcola il coefficiente di Matthews (MCC)
     */
    public static double calcolaMCC(double[] valoriReali, double[] valoriPredetti) {
        double veriPositivi = 0, veriNegativi = 0;
        double falsiPositivi = 0, falsiNegativi = 0;

        for (int i = 0; i < valoriReali.length; i++) {
            boolean realePositivo = (valoriReali[i] == 1.0);
            boolean predettoPositivo = (valoriPredetti[i] == 1.0);

            if (realePositivo && predettoPositivo) {
                veriPositivi++;
            } else if (!realePositivo && !predettoPositivo) {
                veriNegativi++;
            } else if (!realePositivo && predettoPositivo) {
                falsiPositivi++;
            } else if (realePositivo && !predettoPositivo) {
                falsiNegativi++;
            }
        }

        double numeratore = (veriPositivi * veriNegativi) - (falsiPositivi * falsiNegativi);
        double denominatore = Math.sqrt(
            (veriPositivi + falsiPositivi) * 
            (veriPositivi + falsiNegativi) * 
            (veriNegativi + falsiPositivi) * 
            (veriNegativi + falsiNegativi)
        );

        return (denominatore == 0) ? 0.0 : numeratore / denominatore;
    }

    public static void main(String[] args) {
        // Verifica dipendenze
        verificaDipendenze();
        
        long tempoInizio = System.nanoTime();

        try {
            System.out.println("=== Analisi Neuroblastoma con Regressione Lineare ===\n");

            // Carica i dati
            String percorsoFile = "../../data/10_7717_peerj_5665_dataYM2018_neuroblastoma.csv";
            System.out.println("Caricamento dati...");
            Instances dati = caricaDatiDaCSV(percorsoFile);
            System.out.println("✓ File caricato con successo");

            System.out.printf("Dataset originale: %d righe, %d colonne%n", 
                            dati.numInstances(), dati.numAttributes());

            // IMPORTANTE: Rimuovi righe con outcome mancante
            rimuoviRigheConOutcomeMancante(dati);

            // Imputa i valori mancanti nelle caratteristiche
            imputaValoriMancanti(dati);

            int numeroCampioni = dati.numInstances();
            int numeroCaratteristiche = dati.numAttributes() - 1;
            System.out.printf("Dataset finale: %d campioni, %d caratteristiche%n%n", 
                            numeroCampioni, numeroCaratteristiche);

            // Prepara array per predizioni
            double[] predizioni = new double[numeroCampioni];
            double[] esitoReale = dati.attributeToDoubleArray(dati.classIndex());

            // Validazione Leave-One-Out
            System.out.println("Esecuzione validazione Leave-One-Out...");
            System.out.println("(Questo potrebbe richiedere alcuni minuti...)");
            
            for (int i = 0; i < numeroCampioni; i++) {
                // Mostra progresso ogni 10%
                if (i % (numeroCampioni / 10) == 0 && i > 0) {
                    System.out.printf("  Progresso: %.0f%%%n", (i * 100.0 / numeroCampioni));
                }
                
                Instances setAddestramento = new Instances(dati);
                setAddestramento.delete(i);
                Instances setTest = new Instances(dati, i, 1);

                LinearRegression modello = new LinearRegression();
                modello.setAttributeSelectionMethod(new weka.core.SelectedTag(
                    LinearRegression.SELECTION_NONE,
                    LinearRegression.TAGS_SELECTION
                ));
                modello.buildClassifier(setAddestramento);

                predizioni[i] = modello.classifyInstance(setTest.instance(0));
            }
            
            System.out.println("✓ Validazione completata\n");

            // Binarizza le predizioni
            System.out.println("Binarizzazione delle predizioni...");
            double[] predizioniBinarie = new double[numeroCampioni];
            int predettiPositivi = 0, predettiNegativi = 0;
            
            for (int i = 0; i < predizioni.length; i++) {
                predizioniBinarie[i] = (predizioni[i] > 0.5) ? 1.0 : 0.0;
                if (predizioniBinarie[i] == 1.0) predettiPositivi++; else predettiNegativi++;
            }
            
            System.out.printf("  Predizioni: %d positivi, %d negativi%n", predettiPositivi, predettiNegativi);
            
            // Conta valori reali
            int realiPositivi = 0, realiNegativi = 0;
            for (double valore : esitoReale) {
                if (valore > 0.5) realiPositivi++; else realiNegativi++;
            }
            System.out.printf("  Valori reali: %d positivi, %d negativi%n%n", realiPositivi, realiNegativi);

            // Calcola MCC
            double mcc = calcolaMCC(esitoReale, predizioniBinarie);
            
            System.out.println("=".repeat(60));
            System.out.printf("Coefficiente di Correlazione di Matthews (MCC): %.15f%n", mcc);
            System.out.println("=".repeat(60));

            // Tempo di esecuzione
            long tempoFine = System.nanoTime();
            double durataSecondi = (tempoFine - tempoInizio) / 1_000_000_000.0;
            System.out.printf("%n⏱️  Durata dell'esecuzione: %.4f secondi%n", durataSecondi);
            System.out.println("\n✓ Analisi completata con successo!\n");

        } catch (Exception e) {
            System.err.println("✗ Errore durante l'esecuzione: " + e.getMessage());
            e.printStackTrace();
        }
    }
}