packages = c("caret", "mltools")

package.check <- lapply(
  packages,
  FUN = function(x) {
    if (!require(x, character.only = TRUE)) {
      install.packages(x, dependencies = TRUE)
      library(x, character.only = TRUE)
    }
  }
)

start_time <- Sys.time()

data <- read.csv("data/10_7717_peerj_5665_dataYM2018_neuroblastoma.csv", header = TRUE)

for (col_idx in 1:(ncol(data) - 1)) {
  colonna <- data[, col_idx]
  
  # Conta i valori mancanti
  num_mancanti <- sum(is.na(colonna))
  
  if (num_mancanti == 0) {
    next  # Salta se non ci sono valori mancanti
  }
  
  # Estrae i valori non mancanti
  valori_validi <- colonna[!is.na(colonna)]
  
  if (length(valori_validi) == 0) {
    next  # Salta colonne completamente vuote
  }
  
  # Verifica se la colonna è binaria (contiene solo 0 e 1)
  valori_unici <- unique(valori_validi)
  is_binaria <- all(valori_unici %in% c(0, 1))
  
  if (is_binaria) {
    # Colonna binaria: usa MEDIANA
    valore_imputazione <- median(valori_validi)
    tipo_imputazione <- "mediana"
  } else {
    # Colonna reale: usa MEDIA
    valore_imputazione <- mean(valori_validi)
    tipo_imputazione <- "media"
  }
  
  # Sostituisci i valori mancanti
  data[is.na(data[, col_idx]), col_idx] <- valore_imputazione
}

X <- data[, -ncol(data)]
y <- data[, ncol(data)]

ctrl <- trainControl(method = "LOOCV", savePredictions = "final")

model <- train(X, y, method = "lm", trControl = ctrl)

predictions <- model$pred$pred

binary_predictions <- ifelse(predictions > 0.5, 1, 0)

y_ordered <- y[model$pred$rowIndex]

mcc <- mcc(preds = binary_predictions, actuals = y_ordered)
cat(sprintf("Coefficiente di Correlazione di Matthews (MCC): %.15f\n", mcc))

end_time <- Sys.time()
duration <- as.numeric(difftime(end_time, start_time, units = "secs"))
cat(sprintf("Durata dell'esecuzione del programma: %f secondi\n", duration))