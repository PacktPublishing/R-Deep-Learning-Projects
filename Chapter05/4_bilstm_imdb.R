library(purrr)
library(stringr)
library(tm)
library(keras)

df <- read.csv("./data/labeledTrainData.tsv", encoding = "utf-8", quote = "", sep="\t", stringsAsFactors = F)
text <- df$review

corpus <- VCorpus(VectorSource(text))
corpus <- tm_map(corpus,content_transformer(tolower))
corpus <-  tm_map(corpus, content_transformer(removePunctuation))
corpus <-  tm_map(corpus, content_transformer(removeNumbers))
corpus <-  tm_map(corpus, content_transformer(removeWords), stopwords("en"))

dtm <-  DocumentTermMatrix(corpus)
dtm <-  removeSparseTerms(dtm, sparse=0.99)

X <- as.data.frame(as.matrix(dtm))

vocab <- names(X)
maxlen <- 100
dataset <- map(
  1:nrow(X), 
  ~list(review = which(X[.x,]!=0))
)
dataset <- transpose(dataset)

# Vectorization
X <- array(0, dim = c(length(dataset$review), maxlen))
y <- array(0, dim = c(length(dataset$review)))
for(i in 1:length(dataset$review)){
  for(j in 1:maxlen){
    if(length(dataset$review[[i]])>j){
      X[i,j] <- dataset$review[[i]][j]  
    }
    else{
      X[i,j] <- 0
    }
      
  }
  y[i] <- df[i,"sentiment"]
}

X <- as.matrix(X)
X[1,]

# Initialize model
model <- keras_model_sequential()
model %>%
  # Creates dense embedding layer; outputs 3D tensor
  # with shape (batch_size, sequence_length, output_dim)
  layer_embedding(input_dim = length(vocab), 
                  output_dim = 128, 
                  input_length = maxlen) %>% 
  bidirectional(layer_lstm(units = 64)) %>%
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 1, activation = 'sigmoid')

# Compile: you can try different compilers
model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy')
)

# Train model over four epochs
history <- model %>% fit(
  X, y,
  batch_size = 128,
  epochs = 4,
  validation_size = 0.2
)


plot(history)