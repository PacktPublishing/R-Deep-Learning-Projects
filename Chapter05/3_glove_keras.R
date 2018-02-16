library(plyr)
library(dplyr)
library(text2vec)
library(tidytext)
library(caret)

imdb <- read.csv("./data/labeledTrainData.tsv", encoding = "utf-8", quote = "", sep="\t", stringsAsFactors = F)


tokens <- space_tokenizer(imdb$review)
token_iterator <- itoken(tokens)

# Create vocabulary. Terms will be unigrams (simple words).
vocab <- create_vocabulary(token_iterator)

# Reduce non-frequent terms
vocab <- prune_vocabulary(vocab, term_count_min = 5)

# Use our filtered vocabulary
vectorizer <- vocab_vectorizer(vocab)

# use window of 5 for context words
tcm <- create_tcm(token_iterator, vectorizer, skip_grams_window = 5)

###
glove <- GlobalVectors$new(word_vectors_size = 50, 
                           vocabulary = vocab, 
                           x_max = 10)
wv_main <- glove$fit_transform(tcm, 
                              n_iter = 10, 
                              convergence_tol = 0.01)
text <- unlist(imdb$review)
text_df <- data_frame(line = 1:length(text), text = text)

text_df <- text_df %>%
  unnest_tokens(word, text)



wv_context <- glove$components

wv <- as.data.frame(wv_main+t(wv_context))
wv$word <- row.names(wv)

df <-  wv%>% inner_join(text_df)

# Finally create the trained matrix
df <- df %>% group_by(line) %>% summarize_all(mean) %>% select(1:51) 
df$label <- as.factor(imdb$sentiment)


# Create nn in top of it
library(keras)


X <- df[,2:51]
y <- df[,52]

y <- to_categorical(y[["label"]])
y <- y[,2:3]

model <- keras_model_sequential() 

model %>% 
  layer_dense(activation='relu', units =20, input_shape=c(50))%>%
  layer_dense(units=2, activation = 'softmax')

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)


history <- model %>% keras::fit(
  as.matrix(X), y, 
  epochs = 30, batch_size = 128, 
  validation_split = 0.2
)


plot(history)


# Save vector embedding and model
#write.csv(wv,"./data/wv.csv", row.names = F)
#save_model_hdf5(model,"glove_nn.hdf5")
