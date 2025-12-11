#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

double calculate_mcc(const VectorXd& y_true, const VectorXd& y_pred_binary) {
    long long TP = 0, TN = 0, FP = 0, FN = 0;

    for (int i = 0; i < y_true.size(); ++i) {
        bool actual = (y_true(i) == 1.0);
        bool pred = (y_pred_binary(i) == 1.0);

        if (actual && pred) TP++;
        else if (!actual && !pred) TN++;
        else if (!actual && pred) FP++;
        else if (actual && !pred) FN++;
    }

    double numerator = (double)TP * TN - (double)FP * FN;
    double denominator = sqrt(((double)TP + FP) * ((double)TP + FN) * ((double)TN + FP) * ((double)TN + FN));

    if (denominator == 0) return 0.0;
    return numerator / denominator;
}

int main() {
    auto start = std::chrono::high_resolution_clock::now();

    ifstream file("../../data/10_7717_peerj_5665_dataYM2018_neuroblastoma.csv");

    vector<vector<double>> raw_data;
    string line, cell;
    
    while (getline(file, line)) {
        stringstream lineStream(line);
        vector<double> row;
        bool has_nan = false;
        
        while (getline(lineStream, cell, ',')) {
            try {
                double val = stod(cell);
                if (std::isnan(val)) {
                    has_nan = true; 
                    break; 
                }
                row.push_back(val);
            } catch (...) {
                has_nan = true;
                break;
            }
        }
        if (!has_nan && !row.empty()) {
            raw_data.push_back(row);
        }
    }

    if (raw_data.empty()) {
        cerr << "Nessun dato valido trovato." << endl;
        return 1;
    }

    int n_samples = raw_data.size();
    int n_features = raw_data[0].size() - 1;

    MatrixXd X(n_samples, n_features + 1); 
    VectorXd y(n_samples);

    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_features; ++j) {
            X(i, j) = raw_data[i][j];
        }
        X(i, n_features) = 1.0;
        y(i) = raw_data[i][n_features];
    }

    //cout << "--- Matrice X ---" << endl;
    //cout << X << endl << endl; 
    
    //cout << "--- Vettore y ---" << endl;
    //cout << y << endl << endl;

    //cout << "Dimensioni X: " << X.rows() << "x" << X.cols() << endl;
    //cout << "Dimensioni y: " << y.size() << endl;

    VectorXd predictions(n_samples);

    for (int i = 0; i < n_samples; ++i) {
        
        MatrixXd X_train(n_samples - 1, n_features + 1);
        VectorXd y_train(n_samples - 1);
        
        int current_idx = 0;
        for (int j = 0; j < n_samples; ++j) {
            if (i == j) continue;
            X_train.row(current_idx) = X.row(j);
            y_train(current_idx) = y(j);
            current_idx++;
        }

        VectorXd beta = (X_train.transpose() * X_train).ldlt().solve(X_train.transpose() * y_train);

        double pred = X.row(i) * beta;
        predictions(i) = pred;
    }

    VectorXd binary_predictions(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        binary_predictions(i) = (predictions(i) > 0.5) ? 1.0 : 0.0;
    }

    double mcc = calculate_mcc(y, binary_predictions);
    
    cout << "Coefficiente di Correlazione di Matthews (MCC): " << mcc << endl;

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    
    cout << "Durata dell'esecuzione del programma: " << elapsed.count() << " secondi" << endl;

    return 0;
}