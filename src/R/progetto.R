library(caret)

start_time <- Sys.time()

data <- read.csv("GitHub/Tesi-Supervised-ML/data/10_7717_peerj_5665_dataYM2018_neuroblastoma.csv", header = TRUE)
data <- na.omit(data)

X <- data[, -ncol(data)]
y <- data[, ncol(data)]

#print(X)
#print(y)

end_time <- Sys.time()
duration <- as.numeric(difftime(end_time, start_time, units = "secs"))
cat(sprintf("Durata dell'esecuzione del programma: %.10f secondi\n", duration))

X <- data[, -ncol(data)]
y <- data[, ncol(data)]

ctrl <- trainControl(method = "LOOCV")

model <- train(X, y, method = "lm", trControl = ctrl)

predictions <- predict(model, X)

y_pred_binary <- ifelse(predictions > 0.5, 1, 0)

calculate_mcc <- function(y_true, y_pred) {
  TP <- sum(y_true == 1 & y_pred == 1)
  TN <- sum(y_true == 0 & y_pred == 0)
  FP <- sum(y_true == 0 & y_pred == 1)
  FN <- sum(y_true == 1 & y_pred == 0)
  
  numerator <- (TP * TN) - (FP * FN)
  denominator <- sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
  
  if (denominator == 0) return(0.0)
  return(numerator / denominator)
}

mcc <- calculate_mcc(y, y_pred_binary)

cat(sprintf("Coefficiente di Correlazione di Matthews (MCC): %.15f\n", mcc))

end_time <- Sys.time()
duration <- as.numeric(difftime(end_time, start_time, units = "secs"))
cat(sprintf("Durata dell'esecuzione del programma: %.4f secondi\n", duration))