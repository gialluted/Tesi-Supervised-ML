using Pkg

pacchetti_necessari = ["CSV", "DataFrames", "MLJ", "MLJLinearModels", "Statistics"]

for pacchetto in pacchetti_necessari
    try
        eval(Meta.parse("using $pacchetto"))
        println("$pacchetto è già installato")
    catch
        println("$pacchetto non trovato. Installazione in corso...")
        try
            Pkg.add(pacchetto)
            eval(Meta.parse("using $pacchetto"))
            println("$pacchetto installato e caricato con successo")
        catch e
            println("Errore nell'installazione di $pacchetto: $e")
            exit(1)
        end
    end
end

using CSV
using DataFrames
using MLJ
using MLJLinearModels

start_time = time()

data = CSV.read("C:/Users/giall/Documents/GitHub/Tesi-Supervised-ML/data/10_7717_peerj_5665_dataYM2018_neuroblastoma.csv", DataFrame, header=true)

for col_idx in 1:(ncol(data) - 1)
    nome_colonna = names(data)[col_idx]
    colonna = data[:, col_idx]
    
    # Conta i valori mancanti
    num_mancanti = sum(ismissing.(colonna))
    
    if num_mancanti == 0
        continue  # Salta se non ci sono valori mancanti
    end
    
    # Estrae i valori non mancanti
    valori_validi = skipmissing(colonna) |> collect
    
    if isempty(valori_validi)
        continue  # Salta colonne completamente vuote
    end
    
    # Verifica se la colonna è binaria (solo 0 e 1)
    valori_unici = unique(valori_validi)
    is_binaria = all(v -> v == 0 || v == 1, valori_unici)
    
    if is_binaria
        # Colonna binaria: usa MEDIANA
        valore_imputazione = median(valori_validi)
        tipo_imputazione = "mediana"
    else
        # Colonna reale: usa MEDIA
        valore_imputazione = mean(valori_validi)
        tipo_imputazione = "media"
    end
    
    # Sostituisci i valori mancanti
    for row_idx in 1:nrow(data)
        if ismissing(data[row_idx, col_idx])
            data[row_idx, col_idx] = valore_imputazione
        end
    end
end

y = data[:, end]
X = select(data, Not(ncol(data)))
outcome = coerce(y, Continuous)
variabili = coerce(X, Count => Continuous)

# println(X)
# println(y)

LinearRegressor = @load LinearRegressor pkg=MLJLinearModels verbosity=0
model = LinearRegressor()

mach = machine(model, variabili, outcome)
n = length(outcome)
predictions = zeros(n)

for i in 1:n
    train_idx = setdiff(1:n, i)
    fit!(mach, rows=train_idx, verbosity=0)
    predictions[i] = predict(mach, rows=[i])[1]
end

y_pred_class = [pred > 0.5 ? 1 : 0 for pred in predictions]

y_true_cat = coerce(Int.(round.(outcome)), OrderedFactor)
y_pred_cat = coerce(y_pred_class, OrderedFactor)

mcc = matthews_correlation(y_pred_cat, y_true_cat)
println("Coefficiente di Correlazione di Matthews (MCC): $mcc")

durata = time() - start_time
println("Durata dell'esecuzione del programma: $durata secondi")