library(MASS)
library(keras)
Sigma <- matrix(c(1,0,0,1),2,2)
n_points <- 10000
df <- mvrnorm(n=n_points, rep(0,2), Sigma)
df <- as.data.frame(df)

hist(df$V1)
hist(df$V2)
boxplot(df)

# Set the outliers
n_outliers <- as.integer(0.01*n_points)
idxs <- sample(n_points,size = n_outliers)
outliers <- mvrnorm(n=n_outliers, rep(5,2), Sigma)
df[idxs,] <- outliers

plot(df$V1, df$V2, col="red")
points(df[idxs,"V1"], df[idxs,"V2"], col="blue")

boxplot(df)
hist(df$V1)
hist(df$V2)

input_layer <- layer_input(shape=c(2))
encoder <- layer_dense(units=1, activation='relu')(input_layer)
decoder <- layer_dense(units=2)(encoder)
autoencoder <- keras_model(inputs=input_layer, outputs = decoder)

autoencoder %>% compile(optimizer='adam', 
                        loss='mean_squared_error', 
                        metrics=c('accuracy'))

# Coerce the dataframe to matrix to perform the training
df <- as.matrix(df)
history <- autoencoder %>% fit(
  df,df, 
  epochs = 30, batch_size = 128, 
  validation_split = 0.2
)

plot(history)

preds <- autoencoder %>% predict(df)
colnames(preds) <- c("V1", "V2")

df <- as.data.frame(df)
preds <- as.data.frame(preds)


df$color <- ifelse((df$V1-preds$V1)**2+(df$V2-preds$V2)**2>9,"red","blue")

library(ggplot2)
df %>% ggplot(aes(V1,V2),col=df$color)+geom_point(color = df$color, position="jitter")




