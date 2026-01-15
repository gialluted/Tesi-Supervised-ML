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