library(caret)
library(mltools)

start_time <- Sys.time()

data <- read.csv("GitHub/Tesi-Supervised-ML/10_7717_peerj_5665_dataYM2018_neuroblastoma.csv", header = TRUE)
data <- na.omit(data)

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
cat(sprintf("Durata dell'esecuzione del programma: %.4f secondi\n", duration))