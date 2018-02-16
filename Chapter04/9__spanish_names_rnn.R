library(keras)
library(readr)
library(stringr)
library(purrr)
library(tokenizers)


orig <- read_lines("./data/Spanish.txt") 


text <- orig %>%
  str_to_lower() %>%
  str_c(collapse = "\n") %>%
  tokenize_characters(strip_non_alphanum = FALSE, simplify = TRUE)


chars <- text %>% 
         str_c(collapse="\n")%>%
         tokenize_characters(simplify=TRUE) %>% unique %>% sort
chars

max_length <- 5
# Cut the text in semi-redundant sequences of max_length characters
dataset <- map(
  seq(1, length(text) - max_length - 1, by = 3), 
  ~list(name = text[.x:(.x + max_length - 1)], next_char = text[.x + max_length])
)

dataset <- transpose(dataset)

# One-hot vectorization
X <- array(0, dim = c(length(dataset$name), max_length, length(chars)))
y <- array(0, dim = c(length(dataset$name), length(chars)))


for(i in 1:length(dataset$name)){
  X[i,,] <- sapply(chars, function(x){
    as.integer(x == dataset$name[[i]])
  })
  y[i,] <- as.integer(chars == dataset$next_char[[i]])
}

model <- keras_model_sequential()
model %>%
  layer_lstm(128, input_shape = c(max_length, length(chars))) %>%
  layer_dense(length(chars)) %>%
  layer_dropout(0.1)%>%
  layer_activation("softmax")

optimizer <- optimizer_rmsprop(lr = 0.01)
model %>% compile(
  loss = "categorical_crossentropy", 
  optimizer = optimizer
)


sample_mod <- function(preds, temperature = 0.8){
  preds <- log(preds)/temperature
  exp_preds <- exp(preds)
  preds <- exp_preds/sum(exp(preds))
  rmultinom(1, 1, preds) %>% 
    as.integer() %>%
    which.max()
}


history <- model %>% fit(
  X, y,
  batch_size = 128,
  epochs = 20
)
plot(history)


start_idx <- sample(1:(length(text) - max_length), size = 1)
name <- text[start_idx:(start_idx + max_length - 1)]

generated <- ""
for(i in 1:10){
  x <- sapply(chars, function(x){
    as.integer(x == name)
  })
  dim(x) <- c(1, dim(x))
  preds <- predict(model, x)
  next_idx <- sample_mod(preds, 0.3)
  next_char <- chars[next_idx]
  
  generated <- str_c(generated, next_char, collapse = "")
  name <- c(name[-1], next_char)
  cat(generated)
  cat("\n\n")
}


### Generate names of different lengths
## Generate some random names
n_iter <- 100
for(iter in 1:n_iter){
  start_idx <- sample(1:(length(text) - max_length), size = 1)
  name <- text[start_idx:(start_idx + max_length - 1)]
  generated <- " "
  
  random_len <- sample(5:10,1)
  
  for(i in 1:random_len){
    
    x <- sapply(chars, function(x){
      as.integer(x == name)
    })
    dim(x) <- c(1, dim(x))
    
    preds <- predict(model, x)
    next_idx <- sample_mod(preds, 0.1)
    next_char <- chars[next_idx]
    
    generated <- str_c(generated, next_char)
    name <- c(name[-1], next_char)
    #  cat(generated)
    #  cat("\n\n")
  }
  cat(generated)
}

