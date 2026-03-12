import weka.classifiers.functions.LinearRegression;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;

public class proj {

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

    public static void imputaValoriMancanti(Instances dati) {
        int numeroAttributi = dati.numAttributes();
        
        for (int indiceAttr = 0; indiceAttr < numeroAttributi - 1; indiceAttr++) {
            int numeroMancanti = 0;
            
            for (int i = 0; i < dati.numInstances(); i++) {
                if (dati.instance(i).isMissing(indiceAttr)) {
                    numeroMancanti++;
                }
            }
            
            if (numeroMancanti == 0) continue;
            
            boolean eBinaria = eColonnaBinaria(dati, indiceAttr);
            double valorePerImputazione;
            String tipoImputazione;
            
            if (eBinaria) {
                valorePerImputazione = calcolaMediana(dati, indiceAttr);
                tipoImputazione = "mediana";
            } else {
                valorePerImputazione = calcolaMedia(dati, indiceAttr);
                tipoImputazione = "media";
            }
            
            for (int i = 0; i < dati.numInstances(); i++) {
                if (dati.instance(i).isMissing(indiceAttr)) {
                    dati.instance(i).setValue(indiceAttr, valorePerImputazione);
                }
            }
        }
    }

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
        
        long tempoInizio = System.nanoTime();

        String percorsoFile = "../../data/10_7717_peerj_5665_dataYM2018_neuroblastoma.csv";
        Instances dati = caricaDatiDaCSV(percorsoFile);

        imputaValoriMancanti(dati);

        int numeroCampioni = dati.numInstances();
        int numeroCaratteristiche = dati.numAttributes() - 1;

        double[] predizioni = new double[numeroCampioni];
        double[] esitoReale = dati.attributeToDoubleArray(dati.classIndex());
            
        for (int i = 0; i < numeroCampioni; i++) {
                
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
            
        double[] predizioniBinarie = new double[numeroCampioni];
        int predettiPositivi = 0, predettiNegativi = 0;
            
        for (int i = 0; i < predizioni.length; i++) {
            predizioniBinarie[i] = (predizioni[i] > 0.5) ? 1.0 : 0.0;
            if (predizioniBinarie[i] == 1.0) predettiPositivi++; else predettiNegativi++;
        }
            
        int realiPositivi = 0, realiNegativi = 0;
        for (double valore : esitoReale) {
            if (valore > 0.5) realiPositivi++; else realiNegativi++;
        }
            
        double mcc = calcolaMCC(esitoReale, predizioniBinarie);
        System.out.printf("Coefficiente di Correlazione di Matthews (MCC): %.15f%n", mcc);

        long tempoFine = System.nanoTime();
        double durataSecondi = (tempoFine - tempoInizio) / 1_000_000_000.0;
        System.out.printf("%n Durata dell'esecuzione: %.10f secondi%n", durataSecondi);
    }
}