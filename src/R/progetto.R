librerie <- c("caret", "mltools")

for (pacchetto in librerie) {
  if (!require(pacchetto, character.only = TRUE, quietly = TRUE)) {
    cat(sprintf("%s non trovato. Installazione in corso...\n", pacchetto))
    
    tryCatch({
      install.packages(pacchetto, dependencies = TRUE, repos = "https://cran.r-project.org")
      library(pacchetto, character.only = TRUE)
      cat(sprintf("%s installato e caricato con successo\n", pacchetto))
    }, error = function(e) {
      cat(sprintf("Errore nell'installazione di %s: %s\n", pacchetto, conditionMessage(e)))
      quit(status = 1)
    })
  } else {
    cat(sprintf("%s è già installato\n", pacchetto))
  }
}

start_time <- Sys.time()

data <- read.csv("C:/Users/giall/Documents/GitHub/Tesi-Supervised-ML/data/Takashi2019_diabetes_type1_dataset_preprocessed.csv", header = TRUE)

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

variabili <- data[, -ncol(data)]
outcome <- data[, ncol(data)]

ctrl <- trainControl(method = "LOOCV", savePredictions = "final")

model <- train(variabili, outcome, method = "lm", trControl = ctrl)

predictions <- model$pred$pred

binary_predictions <- ifelse(predictions > 0.5, 1, 0)

y_ordered <- outcome[model$pred$rowIndex]

mcc <- mcc(preds = binary_predictions, actuals = y_ordered)
cat(sprintf("Coefficiente di Correlazione di Matthews (MCC): %.15f\n", mcc))

end_time <- Sys.time()
duration <- as.numeric(difftime(end_time, start_time, units = "secs"))
cat(sprintf("Durata dell'esecuzione del programma: %f secondi\n", duration))

flush.console()