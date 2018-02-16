
#http://ai.stanford.edu/~amaas/data/sentiment/

df <- read.csv("./data/labeledTrainData.tsv", encoding = "utf-8", quote = "", sep="\t", stringsAsFactors = F)
text <- df$review

library(tm)
corpus <- VCorpus(VectorSource(text))
inspect(corpus[[1]])

corpus <- tm_map(corpus,content_transformer(tolower))
corpus <- tm_map(corpus, content_transformer(removePunctuation))
corpus <- tm_map(corpus, content_transformer(removeWords), stopwords("en"))
corpus <- tm_map(corpus, stemDocument)


BigramTokenizer <- function(x){ unlist(lapply(ngrams(words(x), 2), paste, collapse = " "), use.names = FALSE)}
dtm <-  DocumentTermMatrix(corpus, control = list(tokenize = BigramTokenizer))
dtm <-  removeSparseTerms(dtm, 0.995)
X <- as.data.frame(as.matrix(dtm))
X$sentiment <- df$sentiment
X$sentiment <- ifelse(X$sentiment<0.5,0,1)

# Train, test, split
library(caTools)
set.seed(42)
spl <-  sample.split(X$sentiment, 0.7)
train <-  subset(X, spl == TRUE)
test <-  subset(X, spl == FALSE)

X_train <- subset(train,select=-sentiment)
y_train <- train$sentiment
X_test <- subset(test,select=-sentiment)
y_test <- test$sentiment

model <- glm(y_train ~ ., data = X_train, family = "binomial")

coefs <- as.data.frame(model$coefficients)
names(coefs) <- c("value")
coefs$token <- row.names(coefs)


library(ggplot2)
library(dplyr)
coefs %>% 
  arrange(desc(value)) %>% 
  head %>% 
  ggplot(aes(x=token, y=value))+
  geom_col()+
  coord_flip()+
  theme_bw()


coefs %>% 
  arrange(value) %>% 
  head %>% 
  ggplot(aes(x=token, y=value))+
  geom_col()+
  coord_flip()+
  theme_bw()



# model performance
roc <- function(y_test, y_preds){
  y_test <- y_test[order(y_preds, decreasing = T)]
  return(data.frame(fpr=cumsum(!y_test)/sum(!y_test),
                    tpr=cumsum(y_test)/sum(y_test)) )
}

y_preds <- predict(model, X_test, type="response")
plot(roc(y_test,y_preds), xlim=c(0,1), ylim=c(0,1))

roc_df <- roc(y_test,y_preds)
ggplot(roc_df, aes(x=fpr,y=tpr))+geom_point(color="red")+theme_bw()


labels <- ifelse(y_preds<0.5,0,1)
table(labels,y_test)
2536/(2536+896)

