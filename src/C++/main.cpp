#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

int main() {
    auto start = std::chrono::high_resolution_clock::now();

    string filename = "../../data/10_7717_peerj_5665_dataYM2018_neuroblastoma.csv";
    ifstream file(filename);
    
    if (!file.is_open()) {
        cerr << "Errore: Impossibile aprire il file " << filename << endl;
        return 1;
    }

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
    
    file.close();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    
    cout << "Durata dell'esecuzione del programma: " << elapsed.count() << " secondi" << endl;

    return 0;
}