df <- read.csv("./data/creditcard.csv", stringsAsFactors = F)
head(df)

library(dplyr)
library(ggplot2)
df %>% ggplot(aes(Time,Amount))+geom_point()+facet_grid(Class~.)

# Look at smaller transactions
df$Class <- as.factor(df$Class)
df %>%filter(Amount<300) %>%ggplot(aes(Class,Amount))+geom_violin()


# Remove the time and class column
idxs <- sample(nrow(df), size=0.1*nrow(df))
train <- df[-idxs,]
test <- df[idxs,]

y_train <- train$Class
y_test <- test$Class

X_train <- train %>% select(-one_of(c("Time","Class")))
X_test <- test %>% select(-one_of(c("Time","Class")))

# Coerce the dataframe to matrix to perform the training
X_train <- as.matrix(X_train)
X_test <- as.matrix(X_test)


## Ok, autoencoder time
library(keras)

input_dim <- 29
outer_layer_dim <- 14
inner_layer_dim <- 7

input_layer <- layer_input(shape=c(input_dim))
encoder <- layer_dense(units=outer_layer_dim, activation='relu')(input_layer)
encoder <- layer_dense(units=inner_layer_dim, activation='relu')(encoder)
decoder <- layer_dense(units=inner_layer_dim)(encoder)
decoder <- layer_dense(units=outer_layer_dim)(decoder)
decoder <- layer_dense(units=input_dim)(decoder)
autoencoder <- keras_model(inputs=input_layer, outputs = decoder)

autoencoder %>% compile(optimizer='adam', 
                        loss='mean_squared_error', 
                        metrics=c('accuracy'))

history <- autoencoder %>% fit(
  X_train,X_train, 
  epochs = 10, batch_size = 32, 
  validation_split=0.2
)

plot(history)

# Reconstruct on the test set
preds <- autoencoder %>% predict(X_test)
preds <- as.data.frame(preds)


y_preds <- ifelse(rowSums((preds-X_test)**2)/30<1,rowSums((preds-X_test)**2)/30,1)



# ROC
# install.packages("ROCR")
library(ROCR)

pred <- prediction(y_preds, y_test)
perf <- performance(pred, measure = "tpr", x.measure = "fpr") 
plot(perf, col=rainbow(10))
