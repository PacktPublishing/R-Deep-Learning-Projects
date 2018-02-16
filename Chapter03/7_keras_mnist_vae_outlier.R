library(keras)

# Switch to the 1-based indexing from R
options(tensorflow.one_based_extract = FALSE)


K <- keras::backend()
mnist <- dataset_mnist()
X_train <- mnist$train$x
y_train <- mnist$train$y
X_test <- mnist$test$x
y_test <- mnist$test$y

## Exclude "0" from the training set. "0" will be the outlier
outlier_idxs <- which(y_train!=0, arr.ind = T)
X_train <- X_train[outlier_idxs,,]
y_test <- sapply(y_test, function(x){ ifelse(x==0,"outlier","normal")})


# reshape
dim(X_train) <- c(nrow(X_train), 784)
dim(X_test) <- c(nrow(X_test), 784)
# rescale
X_train <- X_train / 255
X_test <- X_test / 255

original_dim <- 784
latent_dim <- 2
intermediate_dim <- 256

# Model definition --------------------------------------------------------

X <- layer_input(shape = c(original_dim))
hidden_state <- layer_dense(X, intermediate_dim, activation = "relu")
z_mean <- layer_dense(hidden_state, latent_dim)
z_log_sigma <- layer_dense(hidden_state, latent_dim)

sample_z<- function(params){
  z_mean <- params[,0:1]
  z_log_sigma <- params[,2:3]
  epsilon <- K$random_normal(
    shape = c(K$shape(z_mean)[[1]]), 
    mean=0.,
    stddev=1
  )
  z_mean + K$exp(z_log_sigma/2)*epsilon
}


z <- layer_concatenate(list(z_mean, z_log_sigma)) %>% 
  layer_lambda(sample_z)


# we instantiate these layers separately so as to reuse them later
decoder_hidden_state <- layer_dense(units = intermediate_dim, activation = "relu")
decoder_mean <- layer_dense(units = original_dim, activation = "sigmoid")
hidden_state_decoded <- decoder_hidden_state(z)
X_decoded_mean <- decoder_mean(hidden_state_decoded)

# end-to-end autoencoder
variational_autoencoder <- keras_model(X, X_decoded_mean)

# encoder, from inputs to latent space
encoder <- keras_model(X, z_mean)

# generator, from latent space to reconstructed inputs
decoder_input <- layer_input(shape = latent_dim)
decoded_hidden_state_2 <- decoder_hidden_state(decoder_input)
decoded_X_mean_2 <- decoder_mean(decoded_hidden_state_2)
generator <- keras_model(decoder_input, decoded_X_mean_2)


loss_function <- function(X, decoded_X_mean){
  cross_entropy_loss <- loss_binary_crossentropy(X, decoded_X_mean)
  kl_loss <- -0.5*K$mean(1 + z_log_sigma - K$square(z_mean) - K$exp(z_log_sigma), axis = -1L)
  cross_entropy_loss + kl_loss
}

variational_autoencoder %>% compile(optimizer = "rmsprop", loss = loss_function)
history <- variational_autoencoder %>% fit(
  X_train, X_train, 
  shuffle = TRUE, 
  epochs = 10, 
  batch_size = 256, 
  validation_data = list(X_test, X_test)
)

plot(history)


# Reconstruct on the test set
preds <- variational_autoencoder %>% predict(X_test)
error <- rowSums((preds-X_test)**2)
eval <- data.frame(error=error, class=as.factor(y_test))
library(dplyr)
library(ggplot2)
eval %>% 
  ggplot(aes(x=class,fill=class,y=error))+geom_boxplot()


threshold <- 5
y_preds <- sapply(error, function(x){ifelse(x>threshold,"outlier","normal")})
# Confusion matrix
table(y_preds,y_test)
library(ROCR)
pred <- prediction(error, y_test)
perf <- performance(pred, measure = "tpr", x.measure = "fpr") 
auc <- unlist(performance(pred, measure = "auc")@y.values) 
auc
plot(perf, col=rainbow(10))

