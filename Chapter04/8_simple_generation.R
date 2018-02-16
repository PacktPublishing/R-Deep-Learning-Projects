library(readr)
library(stringr)
library(purrr)
library(tokenizers)
library(dplyr)

orig <- read_lines("./data/female.txt") 

maxlen <- 2

text <- orig %>%
  str_to_lower() %>%
  str_c(collapse = "\n") %>%
  tokenize_characters(strip_non_alphanum = FALSE, simplify = TRUE)

chars <- text %>% unique %>% sort

records <- data.frame()

vec2str <- function(history){
  history <- toString(history)
  history <- str_replace_all(history,",","")
  history <- str_replace_all(history," ","")
  history <- str_replace_all(history,"\n"," ")
  history
}


idxs <- seq(1, length(text) - maxlen - 1, by=100)
for(i in idxs){
  history <-  text[i:(i+maxlen-1)]
  next_char <-  text[i+maxlen]
  history <- vec2str(history)
  records <- rbind(data.frame(history=history, next_char=next_char), records)
  tot_rows <- length(idxs)
}


library(dplyr)
tot_histories <- records %>% group_by(history) %>% summarize(total_h=n())
tot_histories_char <- records %>% group_by(history, next_char) %>% summarize(total_h_c=n())
probas <- left_join(tot_histories, tot_histories_char)
probas$prob <- probas$total_h_c/probas$total_h



generate_next <- function(h){
  sub_df <- probas%>%filter(history==h)
  if(nrow(sub_df)>0){
    prob_vector <- sub_df %>% select(prob)%>%as.matrix %>%c()
    char_vector <- sub_df %>% select(next_char)%>%as.matrix %>%c()
    char_vector <- as.vector(char_vector)
    sample(char_vector,size=1,prob=prob_vector)
  }
}



n_iter <- 10
for(iter in 1:n_iter){
  
  # Generate random initialization
  generated <- " "
  start_index <- sample(1:(length(text) - maxlen), size = 1)
  h <- text[start_index:(start_index + maxlen - 1)]
  h <- vec2str(h)
  
  random_len <- sample(5:10,1)
  
  for(i in 1:random_len){
    c <- generate_next(h)
    h <- paste0(h,c)  
    generated <- str_c(generated,c)
    h <- substr(h,i,i+maxlen)
    
  }
  cat(generated)
  cat("\n")
}
