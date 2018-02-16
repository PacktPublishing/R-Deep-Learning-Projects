library(plyr)
library(dplyr)
library(tidytext)
library(ggplot2)

df <- read.csv("./data/Tweets.csv", stringsAsFactors = F)
text_df <- data_frame(tweet_id=df$tweet_id, tweet=df$text)

text_df <- text_df %>%
  unnest_tokens(word, tweet)


data(stop_words)
head(stop_words)
text_df <- text_df %>% anti_join(stop_words)



# lexicon categorizes words in a binary fashion into positive and negative categories  
bing <- get_sentiments("bing") 

# Join the sentiment data
text_df %>% inner_join(bing)


# Group
text_df %>% inner_join(bing) %>% count(tweet_id,sentiment)

# Plot
text_df %>% 
  inner_join(bing) %>% 
  count(sentiment) %>%
  ggplot(aes(sentiment, n, fill=sentiment))+
  geom_col()+
  theme_bw()



### 
wv <- read.csv("./data/wv.csv")
model <- load_model_hdf5("glove_nn.hdf5")

df <-  wv%>% inner_join(text_df)
head(df)


# Summarize by tweet
df <- df %>% group_by(tweet_id) %>% summarize_all(mean) %>% select(1:51) 
preds <- model %>% predict(as.matrix(df[,2:51]))
hist(preds[,1])
