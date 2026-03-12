#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

bool e_colonna_binaria(const vector<double>& colonna) {
    for (double valore : colonna) {
        if (!isnan(valore) && valore != 0.0 && valore != 1.0) {
            return false;
        }
    }
    return true;
}

double calcola_media(const vector<double>& colonna) {
    double somma = 0.0;
    int conteggio = 0;
    
    for (double valore : colonna) {
        if (!isnan(valore)) {
            somma += valore;
            conteggio++;
        }
    }
    
    return conteggio > 0 ? somma / conteggio : 0.0;
}

double calcola_mediana(const vector<double>& colonna) {
    vector<double> valori_validi;
    
    for (double valore : colonna) {
        if (!isnan(valore)) {
            valori_validi.push_back(valore);
        }
    }
    
    if (valori_validi.empty()) return 0.0;
    
    sort(valori_validi.begin(), valori_validi.end());
    size_t dimensione = valori_validi.size();
    
    if (dimensione % 2 == 0) {
        return (valori_validi[dimensione/2 - 1] + valori_validi[dimensione/2]) / 2.0;
    } else {
        return valori_validi[dimensione/2];
    }
}

double calcola_mcc(const VectorXd& valori_reali, const VectorXd& valori_predetti) {
    long long veri_positivi = 0, veri_negativi = 0;
    long long falsi_positivi = 0, falsi_negativi = 0;

    for (int i = 0; i < valori_reali.size(); ++i) {
        bool reale_positivo = (valori_reali(i) == 1.0);
        bool predetto_positivo = (valori_predetti(i) == 1.0);

        if (reale_positivo && predetto_positivo) {
            veri_positivi++;
        } else if (!reale_positivo && !predetto_positivo) {
            veri_negativi++;
        } else if (!reale_positivo && predetto_positivo) {
            falsi_positivi++;
        } else if (reale_positivo && !predetto_positivo) {
            falsi_negativi++;
        }
    }

    double numeratore = static_cast<double>(veri_positivi * veri_negativi - 
                                            falsi_positivi * falsi_negativi);
    double denominatore = sqrt(
        static_cast<double>((veri_positivi + falsi_positivi) * 
                           (veri_positivi + falsi_negativi) * 
                           (veri_negativi + falsi_positivi) * 
                           (veri_negativi + falsi_negativi))
    );

    return (denominatore == 0.0) ? 0.0 : numeratore / denominatore;
}

int main() {

    auto tempo_inizio = chrono::high_resolution_clock::now();

    const string percorso_file = "C:\\Users\\giall\\Documents\\GitHub\\Tesi-Supervised-ML\\data\\10_7717_peerj_5665_dataYM2018_neuroblastoma.csv";
    
    ifstream file(percorso_file);

    if (!file.is_open()) {
        cerr << "Errore: impossibile aprire il file " << percorso_file << endl;
        return 1;
    }

    vector<vector<double>> dati_grezzi;
    string riga, cella;
    bool prima_riga = true;
    
    while (getline(file, riga)) {
        if (prima_riga) {
            prima_riga = false;
            continue;
        }
        
        stringstream stream_riga(riga);
        vector<double> riga_corrente;
        
        while (getline(stream_riga, cella, ',')) {
            try {
                double valore = stod(cella);
                riga_corrente.push_back(valore);
            } catch (...) {
                riga_corrente.push_back(NAN);
            }
        }
        
        if (!riga_corrente.empty()) {
            dati_grezzi.push_back(riga_corrente);
        }
    }
    
    file.close();

    if (dati_grezzi.empty()) {
        cerr << "Errore: nessun dato trovato nel file." << endl;
        return 1;
    }

    int numero_righe_originale = dati_grezzi.size();
    int numero_colonne = dati_grezzi[0].size();

    int colonna_outcome = numero_colonne - 1;
    vector<vector<double>> dati_puliti;
    int righe_rimosse = 0;
    
    for (const auto& riga : dati_grezzi) {
        if (!isnan(riga[colonna_outcome])) {
            dati_puliti.push_back(riga);
        } else {
            righe_rimosse++;
        }
    }
    
    dati_grezzi = dati_puliti;
    int numero_righe = dati_grezzi.size();

    int mancanti_caratteristiche = 0;
    for (int riga = 0; riga < numero_righe; riga++) {
        for (int col = 0; col < numero_colonne - 1; col++) {
            if (isnan(dati_grezzi[riga][col])) {
                mancanti_caratteristiche++;
            }
        }
    }
    
    for (int col = 0; col < numero_colonne - 1; col++) {
        vector<double> colonna;
        for (int riga = 0; riga < numero_righe; riga++) {
            colonna.push_back(dati_grezzi[riga][col]);
        }
        
        int numero_mancanti = 0;
        for (double valore : colonna) {
            if (isnan(valore)) numero_mancanti++;
        }
        
        if (numero_mancanti == 0) continue;
        
        bool e_binaria = e_colonna_binaria(colonna);
        double valore_per_imputazione;
        string tipo_imputazione;
        
        if (e_binaria) {
            valore_per_imputazione = calcola_mediana(colonna);
            tipo_imputazione = "mediana";
        } else {
            valore_per_imputazione = calcola_media(colonna);
            tipo_imputazione = "media";
        }
        
        for (int riga = 0; riga < numero_righe; riga++) {
            if (isnan(dati_grezzi[riga][col])) {
                dati_grezzi[riga][col] = valore_per_imputazione;
            }
        }
    }

    int numero_campioni = dati_grezzi.size();
    int numero_caratteristiche = numero_colonne - 1;

    MatrixXd X(numero_campioni, numero_caratteristiche + 1);
    VectorXd y(numero_campioni);

    for (int i = 0; i < numero_campioni; ++i) {
        for (int j = 0; j < numero_caratteristiche; ++j) {
            X(i, j) = dati_grezzi[i][j];
        }
        X(i, numero_caratteristiche) = 1.0;
        y(i) = dati_grezzi[i][numero_caratteristiche];
    }

    VectorXd predizioni(numero_campioni);

    for (int i = 0; i < numero_campioni; ++i) {
        MatrixXd X_addestramento(numero_campioni - 1, numero_caratteristiche + 1);
        VectorXd y_addestramento(numero_campioni - 1);
        
        int indice_corrente = 0;
        for (int j = 0; j < numero_campioni; ++j) {
            if (i == j) continue;
            X_addestramento.row(indice_corrente) = X.row(j);
            y_addestramento(indice_corrente) = y(j);
            indice_corrente++;
        }

        VectorXd beta = (X_addestramento.transpose() * X_addestramento)
                        .ldlt()
                        .solve(X_addestramento.transpose() * y_addestramento);

        predizioni(i) = X.row(i).dot(beta);
    }

    VectorXd predizioni_binarie(numero_campioni);
    int predetti_positivi = 0, predetti_negativi = 0;
    
    for (int i = 0; i < numero_campioni; ++i) {
        predizioni_binarie(i) = (predizioni(i) > 0.5) ? 1.0 : 0.0;
        if (predizioni_binarie(i) == 1.0) predetti_positivi++; else predetti_negativi++;
    }

    int reali_positivi = 0, reali_negativi = 0;
    for (int i = 0; i < numero_campioni; ++i) {
        if (y(i) > 0.5) reali_positivi++; else reali_negativi++;
    }

    VectorXd y_binario(numero_campioni);
    for (int i = 0; i < numero_campioni; ++i) {
        y_binario(i) = (y(i) > 0.5) ? 1.0 : 0.0;
    }
    
    double mcc = calcola_mcc(y_binario, predizioni_binarie);
    
    cout << fixed << setprecision(15);
    cout << "Coefficiente di Correlazione di Matthews (MCC): " << mcc << endl;

    auto tempo_fine = chrono::high_resolution_clock::now();
    chrono::duration<double> durata = tempo_fine - tempo_inizio;
    
    cout << "\n Durata dell'esecuzione: " << durata.count() << " secondi" << endl;

    return 0;
}