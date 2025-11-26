// package org.example;

import weka.classifiers.functions.LinearRegression;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import java.io.File;

public class proj {

    public static Instances letturaCSV(String filePath) throws Exception {

        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(filePath));
        Instances data = loader.getDataSet();

        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }

        data.removeIf(Instance::hasMissingValue);

        // System.out.println(data);

        return data;
    }

    public static void main(String[] args) throws Exception {

        long start = System.nanoTime();

        String filePath = "../../data/10_7717_peerj_5665_dataYM2018_neuroblastoma.csv";
        Instances data = letturaCSV(filePath);

        int numSamples = data.numInstances();
        double[] predictions = new double[numSamples];
        double[] outcome = data.attributeToDoubleArray(data.classIndex());

        for (int i = 0; i < numSamples; i++) {
            Instances train = new Instances(data);
            Instances test = new Instances(data, i, 1);

            train.delete(i);

            LinearRegression lr = new LinearRegression();

            lr.setAttributeSelectionMethod(new weka.core.SelectedTag(
                    LinearRegression.SELECTION_NONE,
                    LinearRegression.TAGS_SELECTION
            ));

            lr.buildClassifier(train);

            predictions[i] = lr.classifyInstance(test.instance(0));
        }

        double[] binarizedPredictions = new double[numSamples];
        for (int i = 0; i < predictions.length; i++) {
            if (predictions[i] > 0.5) {
                binarizedPredictions[i] = 1.0;
            } else {
                binarizedPredictions[i] = 0.0;
            }
        }

        double tp = 0;
        double tn = 0;
        double fp = 0;
        double fn = 0;

        for (int i = 0; i < numSamples; i++) {
            boolean actual = (outcome[i] == 1.0);
            boolean predicted = (binarizedPredictions[i] == 1.0);

            if (actual && predicted) {
                tp++;
            } else if (!actual && !predicted) {
                tn++;
            } else if (!actual && predicted) {
                fp++;
            } else if (actual && !predicted) {
                fn++;
            }
        }

        double mccDenominator = Math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn));
        double mcc = (mccDenominator == 0) ? 0.0 : (tp * tn - fp * fn) / mccDenominator;

        System.out.printf("Coefficiente di Correlazione di Matthews (MCC): %.15f%n", mcc);

        long end = System.nanoTime();
        double durationSeconds = (end - start) / 1_000_000_000.0;

        System.out.printf("Durata dell'esecuzione del programma: %.4f secondi%n", durationSeconds);
    }
}