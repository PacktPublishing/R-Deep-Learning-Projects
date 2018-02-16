library(keras)

# Switch to the 1-based indexing from R
options(tensorflow.one_based_extract = FALSE)

K <- keras::backend()
mnist <- dataset_mnist()
X_train <- mnist$train$x
y_train <- mnist$train$y
X_test <- mnist$test$x
y_test <- mnist$test$y
# reshape
dim(X_train) <- c(nrow(X_train), 784)
dim(X_test) <- c(nrow(X_test), 784)
# rescale
X_train <- X_train / 255
X_test <- X_test / 255

K <- keras::backend()

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

# Reconstruction and evaluation on the test set
library(ggplot2)
preds <- variational_autoencoder %>% predict(X_test)
error <- rowSums((preds-X_test)**2)
eval <- data.frame(error=error, class=as.factor(y_test))
eval %>% 
  group_by(class) %>% 
  summarise(avg_error=mean(error)) %>% 
  ggplot(aes(x=class,fill=class,y=avg_error))+geom_col()

# Reshape original and reconstructed
dim(X_test) <- c(nrow(X_test),28,28)
dim(preds) <- c(nrow(preds),28,28)

image(255*preds[1,,], col=gray.colors(3))
y_test[1]
image(255*X_test[1,,], col=gray.colors(3))



# we will sample n points within [-4, 4] standard deviations
grid_x <- seq(-4, 4, length.out = 3)
grid_y <- seq(-4, 4, length.out = 3)

rows <- NULL
for(i in 1:length(grid_x)){
  column <- NULL
  for(j in 1:length(grid_y)){
    z_sample <- matrix(c(grid_x[i], grid_y[j]), ncol = 2)
    column <- rbind(column, predict(generator, z_sample) %>% matrix(ncol = 28) )
  }
  rows <- cbind(rows, column)
}
rows %>% as.raster() %>% plot()
