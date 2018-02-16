library(R6)
set.seed(1234)

softmax <- function(x){
  xt <- exp(x-max(x))
  return(xt/sum(xt))
}
zeros_like <- function(M){
  return(matrix(0,dim(as.matrix(M))[1],dim(as.matrix(M))[2]))  
}
RNN <- R6Class("RNN", 
               public = list(
                 hidden_size = NULL,
                 vocab_size = NULL, 
                 learning_rate = NULL,
                 U = NULL,
                 V = NULL,
                 W = NULL,
                 seq_length = NULL,
                 chars = NULL,
                 n_iter = NULL,
                 initialize = function(hidden_size = NA, vocab_size = NA, chars=NA, n_iter=100, seq_length=NA, learning_rate=0.01){
                   self$hidden_size <- hidden_size
                   self$n_iter <- n_iter
                   self$learning_rate <- learning_rate
                   self$seq_length <- seq_length
                   self$vocab_size <- as.numeric(vocab_size)
                   self$chars <- chars
                   self$U <- matrix(rnorm(hidden_size*vocab_size)*0.01, nrow=self$hidden_size) # input to hidden
                   self$W <-  matrix(rnorm(hidden_size*hidden_size)*0.01, nrow=self$hidden_size) # hidden to hidden
                   self$V <-  matrix(rnorm(vocab_size*hidden_size)*0.01, nrow=self$vocab_size) # hidden to output
                 }
                 , forward_step = function(input_sample){
                      ## Takes one column vector and returns probabilities
                   x <- input_sample
                   s <- tanh(self$U%*%x+self$W%*%self$s)
                   o <- softmax(self$V%*%self$s)
                   return(list("pred"=o,"state"=s))
                 }
                 , bptt = function(inputs,targets,s_prev){
                   seq_size <- length(inputs) #total length of the sequence
                   xs <- lapply(vector('list',seq_size), function(i) matrix(0,self$vocab_size, 1))
                   hs <- lapply(vector('list',seq_size), function(i) matrix(0,self$hidden_size, 1))
                   ys <- lapply(vector('list',seq_size), function(i) matrix(0,self$vocab_size, 1))
                   ps <- lapply(vector('list',seq_size), function(i) matrix(0,self$vocab_size,1))
                   loss <- 0
                   for(idx in 1:seq_size){
                     xs[[idx]] <- matrix(0,self$vocab_size,1)
                     xs[[idx]][inputs[[idx]]] = 1
                     ## Update the hidden state
                     if(idx==1){
                       hs[[idx]] <- tanh(self$U%*%xs[[idx]]+self$W%*%s_prev)  
                     }
                     else{
                       hs[[idx]] <- tanh(self$U%*%xs[[idx]]+self$W%*%hs[[(idx-1)]])  
                     }
                     ## calculating probabilities for the next character
                     ys[[idx]] = self$V%*%hs[[idx]]
                     ps[[idx]] = softmax(ys[[idx]])
                     
                     ## Cross-entropy loss
                     loss = loss-log(ps[[idx]][targets[idx], 1])
                     
                     # Calculate gradients
                     dU <- zeros_like(self$U)
                     dW <- zeros_like(self$W)
                     dV <- zeros_like(self$V)
                     dhnext <-  zeros_like(s_prev)
                     for(j in length(inputs):1){
                       ## Gradient of the error vs output
                       dy <- ps[[j]]
                       dy[targets[j]] <- dy[targets[j]]-1 
                       dV <- dV+dy%*%t(hs[[j]])
                       dh <- t(self$V)%*%dy + dhnext
                       ## backprop through tanh nonlinearity
                       dhraw <-  (1 - hs[[j]] * hs[[j]]) * dh
                       ## derivative of the error between input and hidden layer
                       dU <- dU+dhraw%*%t(xs[[j]])
                       if(j==1){
                         dW <- dW+dhraw%*%t(s_prev) 
                       }
                       else{
                         dW <- dW+dhraw%*%t(hs[[(j-1)]])   
                       }
                      dhnext <- t(self$W)%*%dhraw
                     }
                   }
                   return(list("loss"=loss, "dU"=dU, "dW"=dW, "dV"=dV, "hs"=hs[length(inputs)-1]))
                 }
                 , 
                 sample_char = function(h, seed_ix, n){
                   #Generate a sequence of characters given a seed and a hidden state
                   x <-  matrix(0,self$vocab_size, 1)
                   x[seed_ix] <-  1
                   ixes <- c()
                   for(t in 1:n){
                     h <- tanh(self$U%*%x+self$W%*%h)
                     y <- self$V%*%h
                     p <- exp(y)/sum(exp(y)) #softmax
                     ix <- sample(self$chars,size=1, replace=T, prob=p)
                     x <- matrix(0,self$vocab_size,1)
                     x[which(chars==ix)] <- 1
                     ixes[t] <- ix
                   }
                   return(ixes)
                 }
                 , train = function(text){
                   n <-  1
                   p <-  1
                   mU <- zeros_like(self$U)
                   mW <- zeros_like(self$W)
                   mV <-  zeros_like(self$V)
                   
                   # memory variables for Adagrad
                   smooth_loss = -log(1.0/self$vocab_size)*self$seq_length # loss at iteration 0
                   
                   for(n in 1:self$n_iter){
                     # 
                     if(p + self$seq_length + 1 >= length(text) || n == 1){
                       # reset RNN memory
                       ## h_old is the previous hiddden state of RNN
                       h_old <- matrix(0,self$hidden_size, 1)
                       # go from start of data
                       p <-  1
                     }
                     inputs <-  unlist(lapply(text[p:(p+self$seq_length)],function(c){which(self$chars==c)}))
                     targets <- unlist(lapply(text[(p+1):(p+self$seq_length+1)],function(c){which(self$chars==c)}))
                     # sample from the model now and then
                     if(n %% 100 == 0){
                       txt <-  self$sample_char(h_old, inputs[[1]], 200)
                       ## Find the line breaks 
                       line_breaks <- which(txt=="\n")
                       if(length(line_breaks)<2){
                         print(txt)
                       }
                       else{
                         for(ix in 2:(length(line_breaks-1))){
                           first_ix <- line_breaks[ix-1]+1
                           last_ix <- line_breaks[ix]-1
                           print(paste(txt[first_ix:last_ix], collapse=""))
                         }
                       }
                       smooth_loss = smooth_loss*0.999+loss*0.001
                       print('---- sample -----')
                       cat("Iteration number: ",n, "\n")
                       cat("Loss: ", smooth_loss)
                     }
                     tmp <-  self$bptt(inputs, targets, h_old)
                     loss <- unlist(tmp$loss)
                     dU <- unlist(tmp$dU)
                     dW <- unlist(tmp$dW) 
                     dV <- unlist(tmp$dV)
                     h_old <- unlist(tmp$hs)
                     ## Time to update the Adagrad weights
                     # U
                     mU <- mU+dU**2
                     self$U <- self$U-self$learning_rate * dU  / sqrt(mU + 1e-8)
                     # W
                     mW <- mW+dW**2
                     self$W <- self$W-self$learning_rate * dW / sqrt(mW + 1e-8)
                    # V
                     mV <- mV+dV**2
                     self$V <- self$V-self$learning_rate * dV / sqrt(mV + 1e-8)
                     p <- p+self$seq_length
                     n <- n+1
                   }
                   return(1)
                 }
    )
  )




################################################
############### USE FOR TEXT####################
################################################

library(readr)
library(stringr)
library(purrr)
library(tokenizers)
data <- read_lines("./data/female.txt")
text <- data %>%
  str_to_lower() %>%
  str_c(collapse = "\n") %>%
  tokenize_characters(strip_non_alphanum = FALSE, simplify = TRUE)
chars <- text %>% unique

test <- RNN$new(hidden_size = 100, 
                vocab_size = length(chars), 
                chars=chars, 
                n_iter=100, seq_length=25,
                learning_rate=0.01)
test$train(text)



