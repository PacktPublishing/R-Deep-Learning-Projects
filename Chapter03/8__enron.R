install.packages("tm")
install.packages("SnowballC")

df <- read.csv("./data/enron.csv")
names(df)

library(tm)
corpus <- Corpus(VectorSource(df$email))
inspect(corpus[[1]])

corpus <- tm_map(corpus,tolower)
corpus <-  tm_map(corpus, removePunctuation)
corpus <-  tm_map(corpus, removeWords, stopwords("english"))
corpus <-  tm_map(corpus, stemDocument)

dtm <-  DocumentTermMatrix(corpus)
dtm <-  removeSparseTerms(dtm, 0.97)


X <- as.data.frame(as.matrix(dtm))
X$responsive <- df$responsive


# Train, test, split
library(caTools)
set.seed(42)
spl <-  sample.split(X$responsive, 0.7)
train <-  subset(X, spl == TRUE)
test <-  subset(X, spl == FALSE)
train <- subset(train, responsive==0)

X_train <- subset(train,select=-responsive)
y_train <- train$responsive
X_test <- subset(test,select=-responsive)
y_test <- test$responsive


library(keras)
input_dim <- ncol(X_train)
inner_layer_dim <- 32
input_layer <- layer_input(shape=c(input_dim))
encoder <- layer_dense(units=inner_layer_dim, activation='relu')(input_layer)
decoder <- layer_dense(units=input_dim)(encoder)
autoencoder <- keras_model(inputs=input_layer, outputs = decoder)
autoencoder %>% compile(optimizer='adam', 
                        loss='mean_squared_error', 
                        metrics=c('accuracy'))


X_train <- as.matrix(X_train)
X_test <- as.matrix(X_test)
history <- autoencoder %>% fit(
  X_train,X_train, 
  epochs = 100, batch_size = 32, 
  validation_data = list(X_test, X_test)
)

plot(history)

# Reconstruct on the test set
preds <- autoencoder %>% predict(X_test)
error <- rowSums((preds-X_test)**2)
# install.packages("tidyverse")
library(tidyverse)
eval %>% 
  filter(error < 1000) %>%
  ggplot(aes(x=error,color=class))+geom_density()




threshold <- 100
y_preds <- sapply(error, function(x){ifelse(x>threshold,"outlier","normal")})
# Confusion matrix
table(y_preds,y_test)

library(ROCR)
pred <- prediction(error, y_test)
perf <- performance(pred, measure = "tpr", x.measure = "fpr") 
auc <- unlist(performance(pred, measure = "auc")@y.values) 
auc
plot(perf, col=rainbow(10))
