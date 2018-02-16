library(h2o)
h2o.init()
df <- read.csv("./data/creditcard.csv", stringsAsFactors = F)
df <- as.h2o(df)


splits <- h2o.splitFrame(df, ratios=c(0.8), seed=1)
train <- splits[[1]]
test <- splits[[2]]

label <- "Class"
features <- setdiff(colnames(train), label)

autoencoder <- h2o.deeplearning(x=features, 
                                training_frame = train, 
                                autoencoder = TRUE, 
                                seed = 1, 
                                hidden=c(10,2,10), 
                                epochs = 10,
                                activation = "Tanh")


# Use the predict function as before
preds <- h2o.predict(autoencoder, test)

library(tidyverse)

anomaly <- h2o.anomaly(autoencoder, test) %>%
  as.data.frame() %>%
  tibble::rownames_to_column() %>%
  mutate(Class = as.vector(test[, 31]))

anomaly$Class <- as.factor(anomaly$Class)

mean_mse <- anomaly %>%
  group_by(Class) %>%
  summarise(mean = mean(Reconstruction.MSE))


ggplot(anomaly, aes(x = as.numeric(rowname), y = Reconstruction.MSE, color = Class)) +
  geom_point() +
  geom_hline(data = mean_mse, aes(yintercept = mean, color = Class)) +
  scale_color_brewer(palette = "Set2") +
  labs(x = "instance number",
       color = "Class")

## TO DO: Improve this model using supervised learning... recycle the features from the encoding
