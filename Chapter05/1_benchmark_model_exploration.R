
#install.packages("tidytext")

library(tidytext)
library(dplyr)


# Tidy data: (Wickham, Hadley. 2014. “Tidy Data.” Journal of Statistical Software 59 (1): 1–23. doi:10.18637/jss.v059.i10.)
#Each variable is a column
#Each observation is a row
#Each type of observational unit is a table


text <- c("The food is typical Czech, and the beer is good. The service is quick, if short and blunt, and the waiting on staff could do with a bit of customer service training",
          "The food was okay. Really not bad, but we had better",
          "A venue full of locals. No nonsense, no gimmicks. Only went for drinks which were good and cheap. People friendly enough.",
          "Great food, lovely staff, very reasonable prices considering the location!")
text_df <- data_frame(line = 1:4, text = text)


text_df <- text_df %>%
  unnest_tokens(word, text)

data(stop_words)
head(stop_words)
text_df <- text_df %>% anti_join(stop_words)
head(text_df)


library(ggplot2)
text_df %>% 
  count(word, sort=T) %>%
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(word, n)) +
  geom_col() +
  xlab(NULL) +
  coord_flip()+
  theme_bw()


############ Different lexicon #############

get_sentiments("afinn") %>% 
  filter(score==-5) %>% 
  head

get_sentiments("afinn") %>% 
  filter(score==0) %>% 
  head

get_sentiments("afinn") %>% 
  filter(score==5) %>% 
  head


get_sentiments("bing") %>% head


get_sentiments("nrc") %>% head


############################# Bing sentiment dat ###################################

# lexicon categorizes words in a binary fashion into positive and negative categories  
bing <- get_sentiments("bing") # https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html (Bing Liu and coauthors)

# Join the sentiment data
text_df %>% inner_join(bing)


# Group
text_df %>% inner_join(bing) %>% count(line,sentiment)

# Plot
text_df %>% 
  inner_join(bing) %>% 
  count(line,sentiment) %>%
  ggplot(aes(line, n, fill=sentiment))+
  geom_col()+
  coord_flip()+
  theme_bw()


############################# Afinn sentiment dat ###################################
afinn <- get_sentiments("afinn")


# Join the sentiment data
text_df %>% inner_join(afinn)


# Group
text_df %>% 
  inner_join(afinn) %>% 
  group_by(line) %>% 
  summarize(total_score = sum(score))

# Plot
text_df %>% 
  inner_join(afinn) %>% 
  group_by(line) %>% 
  summarize(total_score = sum(score)) %>%
  mutate(sentiment=ifelse(total_score>0,"positive","negative")) %>%
  ggplot(aes(line, total_score, fill=sentiment))+
  geom_col()+
  coord_flip()+
  theme_bw()


# What happened here? Review 3 is mostly positive

################################### Using bigrams  ###################################


text_df <- data_frame(line = 1:4, text = text)
text_df <- text_df %>%
  unnest_tokens(bigram, text, token="ngrams", n=2)
text_df


library(tidyr)
text_df <- text_df %>% separate(bigram, c("w1","w2"), sep=" ")
text_df


text_df %>% 
  filter(w1=="no") %>% 
  inner_join(afinn, by=c(w2="word"))

