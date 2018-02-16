library(keras)
mnist <- dataset_mnist()

X_train <- mnist$train$x
y_train <- mnist$train$y
X_test <- mnist$test$x
y_test <- mnist$test$y

## Exclude "7" from the training set. "7" will be the outlier
outlier_idxs <- which(y_train!=7, arr.ind = T)
X_train <- X_train[outlier_idxs,,]
y_test <- sapply(y_test, function(x){ ifelse(x==7,"outlier","normal")})

# The x data is a 3-d array (images,width,height) 
# of grayscale values . To prepare the data for 
# training we convert the 3-d arrays into matrices 
# by reshaping width and height into a single dimension 
# (28x28 images are flattened into length 784 vectors). 
# Then, we convert the grayscale values from integers 
# ranging between 0 to 255 into floating point values 
# ranging between 0 and 1:

# reshape
dim(X_train) <- c(nrow(X_train), 784)
dim(X_test) <- c(nrow(X_test), 784)
# rescale
X_train <- X_train / 255
X_test <- X_test / 255
input_dim <- 28*28 #784
inner_layer_dim <- 32

# Create the autoencoder
input_layer <- layer_input(shape=c(input_dim))
encoder <- layer_dense(units=inner_layer_dim, activation='relu')(input_layer)
decoder <- layer_dense(units=784)(encoder)
autoencoder <- keras_model(inputs=input_layer, outputs = decoder)

autoencoder %>% compile(optimizer='adam', 
                        loss='mean_squared_error', 
                        metrics=c('accuracy'))

history <- autoencoder %>% fit(
  X_train,X_train, 
  epochs = 50, batch_size = 256, 
  validation_split=0.2
)
plot(history)

# Reconstruct on the test set
preds <- autoencoder %>% predict(X_test)
error <- rowSums((preds-X_test)**2)

eval <- data.frame(error=error, class=as.factor(y_test))
library(ggplot2)
library(dplyr)
eval %>% 
  group_by(class) %>% 
  summarise(avg_error=mean(error)) %>% 
  ggplot(aes(x=class,fill=class,y=avg_error))+geom_boxplot()


threshold <- 15
y_preds <- sapply(error, function(x){ifelse(x>threshold,"outlier","normal")})

# Confusion matrix
table(y_preds,y_test)
  
