library(readr)
library(stringr)
library(purrr)
library(tokenizers)

"
Minimal character-level Vanilla RNN model. 
R version by Pablo Maldonado
Original version by Andrej Karpathy (@karpathy) 
BSD License
"


set.seed(1234)


zeros_like <- function(M){
  return(matrix(0,dim(as.matrix(M))[1],dim(as.matrix(M))[2]))  
}


softmax <- function(x){
  xt <- exp(x-max(x))
  return(xt/sum(xt))
}

data <- read_lines("C:/Users/test/Desktop/8604/Chapter04/female.txt")
head(data)


text <- data %>%
  str_to_lower() %>%
  str_c(collapse = "\n") %>%
  tokenize_characters(strip_non_alphanum = FALSE, simplify = TRUE)


chars <- text %>% unique
chars


# hyperparameters
hidden_size = 100 # size of hidden layer of neurons
seq_length = 10 # number of steps to unroll the RNN for
learning_rate = 1e-1
vocab_size = length(chars)


U <- matrix(rnorm(hidden_size*vocab_size)*0.01, nrow=hidden_size) # input to hidden
W <-  matrix(rnorm(hidden_size*hidden_size)*0.01, nrow=hidden_size) # hidden to hidden
V <-  matrix(rnorm(vocab_size*hidden_size)*0.01, nrow=vocab_size) # hidden to output
bh <-  matrix(0,hidden_size, 1) # hidden bias
by <-  matrix(0,vocab_size, 1) # output bias


lossFun <- function(inputs,targets,prev_hidden){
  tot <- length(inputs) #total sequence length
  xs <- lapply(vector('list',tot), function(i) matrix(0,vocab_size, 1))
  hs <- lapply(vector('list',tot), function(i) matrix(0,hidden_size, 1))
  ys <- lapply(vector('list',tot), function(i) matrix(0,vocab_size, 1))
  ps <- lapply(vector('list',tot), function(i) matrix(0,vocab_size,1))
  loss <- 0

  for(idx in 1:tot){
    xs[[idx]] <- matrix(0,vocab_size,1)
    xs[[idx]][inputs[[idx]]] = 1

    ## update the hidden state
    if(idx==1){
      hs[[idx]] <- tanh(U%*%xs[[idx]]+W%*%h_old+bh)  
    }
    else{
      hs[[idx]] <- tanh(U%*%xs[[idx]]+W%*%hs[[(idx-1)]]+bh)  
    }
    
    ## Get char probabilities
    ys[[idx]] = V%*%hs[[idx]] + by
    ps[[idx]] = softmax(ys[[idx]])
    
    ## Loss function (cross-entropy here)
    loss <-  loss-log(ps[[idx]][targets[idx], 1])
    
    # Initialize the gradients
    dU <- zeros_like(U)
    dW <- zeros_like(W)
    dV <- zeros_like(V)
    dbh <- zeros_like(bh)
    dby <- zeros_like(by)
    dhnext <-  zeros_like(h_old)
    
    # Here comes the backprop loop
    for(j in length(inputs):1){
      # Output vs loss
      dy <- ps[[j]]
      dy[targets[j]] <- dy[targets[j]]-1 
      dV <- dV+dy%*%t(hs[[j]])
      dby <-  dby+dy
      
      ## Hidden layer
      dh <-  t(V)%*%dy + dhnext
      dh_raw <-  (1 - hs[[j]] * hs[[j]]) * dh
      dbh <- dbh+dh_raw
      
      dU <- dU+dh_raw%*%t(xs[[j]])
      
      if(j==1){
        dW <- dW+dh_raw%*%t(h_old) 
      }
      else{
        dW <- dW+dh_raw%*%t(hs[[(j-1)]])   
      }
      dhnext <- t(W)%*%dh_raw
      
      
    }
  }
  return(list("loss"=loss, "dU"=dU, "dW"=dW, "dV"=dV, "dbh"=dbh, "dby"=dby, "hs"=hs[length(inputs)-1]))
}


## Sample a few chars given a hidden state and a seed
sample_char <- function(h, seed_ix, n){
  x <-  matrix(0,vocab_size, 1)
  x[seed_ix] <-  1
  
  ixes <- c()
  
  for(t in 1:n){
    h <- tanh(U%*%x+W%*%h+bh)
    y <- V%*%h+by
    p <- exp(y)/sum(exp(y)) #softmax
    ix <- sample(chars,size=1, replace=T, prob=p)
    x <- matrix(0,vocab_size,1)
    x[which(chars==ix)] <- 1
    ixes[t] <- ix
  }
  return(ixes)
}


n <-  1
p <-  1

mU <- zeros_like(U)
mW <- zeros_like(W)
mV <-  zeros_like(V)
mbh <-  zeros_like(bh)
mby <- zeros_like(by) # memory variables for Adagrad
smooth_loss = -log(1.0/vocab_size)*seq_length # loss at iteration 0

while(T){
  if(p + seq_length + 1 >= length(data) || n == 1){
    # reset RNN memory
    ## h_old is the hiddden state of RNN
    h_old <- matrix(0,hidden_size, 1)
    # go from the start of the data
    p <-  1
  }
  
  inputs <-  unlist(sapply(text[p:(p+seq_length)],function(c){which(chars==c)}))
  targets <- unlist(sapply(text[(p+1):(p+seq_length+1)],function(c){which(chars==c)}))
  
  # Check what the model is doing from time to time
  if(n %% 100 == 0){
    txt <-  sample_char(h_old, inputs[[1]], 200)
    ## Find line breaks
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
  
  tmp <-  lossFun(inputs, targets, h_old)
  loss <- unlist(tmp$loss)
  dU <- unlist(tmp$dU)
  dW <- unlist(tmp$dW) 
  dV <- unlist(tmp$dV)
  dbh <- unlist(tmp$dbh)
  dby <- unlist(tmp$dby)
  h_old <- unlist(tmp$hs)
  
  ## Adagrad updates
  
  # U
  mU <- mU+dU**2
  U <- U-learning_rate * dU  / sqrt(mU + 1e-8)
    
  # W
  mW <- mW+dW**2
  W <- W-learning_rate * dW / sqrt(mW + 1e-8)
    
  # V
  mV <- mV+dV**2
  V <- V-learning_rate * dV / sqrt(mV + 1e-8)
  
  # bh
  mbh <- mbh+mbh**2
  bh <- bh-learning_rate * dbh / sqrt(mbh + 1e-8)
  
  # by
  mby <- mby+dby**2
  by <- by-learning_rate * dby / sqrt(mby + 1e-8)
 
  p <- p+seq_length
  n <- n+1
}
  