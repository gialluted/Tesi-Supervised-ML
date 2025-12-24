using CSV
using DataFrames
using MLJ
using MLJLinearModels

start_time = time()

df = CSV.read("data/10_7717_peerj_5665_dataYM2018_neuroblastoma.csv", DataFrame, header=true)
dropmissing!(df)

y_raw = df[:, end]
X_raw = select(df, Not(ncol(df)))
y = coerce(y_raw, Continuous)
X = coerce(X_raw, Count => Continuous)

# println(X)
# println(y)

LinearRegressor = @load LinearRegressor pkg=MLJLinearModels verbosity=0
model = LinearRegressor()

mach = machine(model, X, y)
n = length(y)
predictions = zeros(n)

for i in 1:n
    train_idx = setdiff(1:n, i)
    fit!(mach, rows=train_idx, verbosity=0)
    predictions[i] = predict(mach, rows=[i])[1]
end

y_pred_class = [pred > 0.5 ? 1 : 0 for pred in predictions]

y_true_cat = coerce(Int.(round.(y)), OrderedFactor)
y_pred_cat = coerce(y_pred_class, OrderedFactor)

mcc = matthews_correlation(y_pred_cat, y_true_cat)
println("Coefficiente di Correlazione di Matthews (MCC): $mcc")

durata = time() - start_time
println("Durata dell'esecuzione del programma: $durata secondi")