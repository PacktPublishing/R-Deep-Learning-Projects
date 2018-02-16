library(MASS)
library(R6)
set.seed(1234)

softmax <- function(x){
  xt <- exp(x-max(x))
  return(xt/sum(xt))
}

zeros_like <- function(M){
  return(matrix(0,dim(as.matrix(M))[1],dim(as.matrix(M))[2]))  
}

cem <- function(f, theta_mean, theta_std, n_iter=300, batch_size=25, elite_frac=0.2){
  # Now, for the algorithms
  for(it in 1:n_iter){
    # Sample parameter vectors
    thetas <-  matrix(mvrnorm(n=batch_size*dim_theta, mu= theta_mean, Sigma=theta_std), ncol = dim_theta)
    rewards <- apply(thetas,1,f) 
    # Get elite parameters
    n_elite <-  as.integer(batch_size * elite_frac)
    elite_inds <-  sort(rewards, decreasing = T, index.return=T)$ix[1:n_elite]
    elite_thetas <- thetas[elite_inds,]
    # Update theta_mean, theta_std
    theta_mean <- apply(elite_thetas, 2,mean)
    theta_std <- 0.01*diag(dim_theta)+0.99*cov(elite_thetas)
  }
  return(theta_mean)
}  

CEM <- R6Class("RNN-CEM", 
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
                      ## Takes one column vector and returns the softmax output
                   x <- input_sample
                   s <- tanh(self$U%*%x+self$W%*%self$s)
                   o <- softmax(self$V%*%self$s)
                   return(list("pred"=o,"state"=s))
                 }
                 , forward = function(theta, inputs,targets,s_prev){
                   
                   U <- as.matrix(theta[1:hidden_size*vocab_size],nrow=self$hidden_size)
                   W <-  as.matrix(theta[(hidden_size*vocab_size+1):(hidden_size*(vocab_size+hidden_size)+1)],
                                        nrow=self$hidden_size)
                   V <-  as.matrix(theta[(hidden_size*(vocab_size+hidden_size)+1):length(theta_m)], nrow=self$vocab_size) # hidden to output
                   
                   seq_size <- length(inputs) #total length of the sequence
                   xs <- lapply(vector('list',seq_size), function(i) matrix(0,self$vocab_size, 1))
                   hs <- lapply(vector('list',seq_size), function(i) matrix(0,self$hidden_size, 1))
                   ys <- lapply(vector('list',seq_size), function(i) matrix(0,self$vocab_size, 1))
                   ps <- lapply(vector('list',seq_size), function(i) matrix(0,self$vocab_size,1))
                   loss <- 0
                   for(idx in 1:seq_size){
                     xs[[idx]] <- matrix(0,self$vocab_size,1)
                     xs[[idx]][inputs[[idx]]] = 1
                     ## hidden state, using previous hidden state hs[t-1]
                     if(idx==1){
                       hs[[idx]] <- tanh(U%*%xs[[idx]]+W%*%s_prev)  
                     }
                     else{
                       hs[[idx]] <- tanh(U%*%xs[[idx]]+W%*%hs[[(idx-1)]])  
                     }
                     ## unnormalized log probabilities for next chars
                     ys[[idx]] = V%*%hs[[idx]]
                     ## probabilities for next chars, softmax
                     ps[[idx]] = softmax(ys[[idx]])
                     ## Cross-entropy loss
                     loss = loss-log(ps[[idx]][targets[idx], 1])
                   }
                   return(loss)
                 }
                 
                 , ## given a hidden RNN state, and a input char id, predict the coming n chars
                 sample_char = function(h, seed_ix, n){
                   "  
                   sample a sequence of integers from the model
                   h is memory state, seed_ix is seed letter for first time step
                   "
                   ## a one-hot vector
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
                   ## iterator counter
                   n <-  1
                   ## data pointer
                   p <-  1
                   
                   smooth_loss = -log(1.0/self$vocab_size)*self$seq_length # loss at iteration 0
                   
                   for(n in 1:self$n_iter){
                     # prepare inputs (we're sweeping from left to right in steps seq_length long)
                     if(p + self$seq_length + 1 >= length(text) || n == 1){
                       # reset RNN memory
                       ## s_prev is the hiddden state of RNN
                       s_prev <- matrix(0,self$hidden_size, 1)
                       # go from start of data
                       p <-  1
                     }
                     inputs <-  unlist(lapply(text[p:(p+self$seq_length)],function(c){which(self$chars==c)}))
                     targets <- unlist(lapply(text[(p+1):(p+self$seq_length+1)],function(c){which(self$chars==c)}))
                     # sample from the model now and then
                     if(n %% 100 == 0){
                       txt <-  self$sample_char(s_prev, inputs[[1]], 200)
                       ## Find the \n in the string
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
                     ## UPDATES
                     theta_m <- c(as.vector(self$U), as.vector(self$V), as.vector(self$W))
                     new_m <- cem(-self$forward, theta_m, diag(length(theta_m))*0.01) 
                     
                     self$U <- as.matrix(theta_m[1:hidden_size*vocab_size],nrow=self$hidden_size)
                     self$W <-  as.matrix(theta_m[(hidden_size*vocab_size+1):(hidden_size*(vocab_size+hidden_size)+1)],
                                          nrow=self$hidden_size)
                     self$V <-  as.matrix(theta_m[(hidden_size*(vocab_size+hidden_size)+1):length(theta_m)], nrow=self$vocab_size) # hidden to output
                     
                     loss <- self$forward(inputs,targets,s_prev)
                     p <- p+self$seq_length
                     n <- n+1
                   }
                   return(1)
                 }
    )
  )


# Now let's generate some names!
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
test <- CEM$new(hidden_size = 100, 
                vocab_size = length(chars), 
                chars=chars, 
                n_iter=100, seq_length=25,
                learning_rate=0.01)
test$train(text)



