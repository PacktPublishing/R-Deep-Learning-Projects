library(R6)

sigmoid <- function(x){
  1/(1+exp(-x))
}

MLP <- R6Class("MLP", 
                      public = list(
                        dim = NULL,
                        n_iter = NULL,
                        learning_rate = NULL,
                        hidden_layer_size=NULL,
                        Wih = NULL,
                        Who = NULL,
                        a = NULL,
                        initialize = function(learning_rate = 0.3, n_iter=NA, dim=NA, hidden_layer_size=NA){
                          self$dim <- dim
                          self$n_iter <- n_iter
                          self$learning_rate <- learning_rate
                          self$hidden_layer_size <- hidden_layer_size
                          self$Wih <- matrix(runif(self$hidden_layer_size*self$dim), ncol = self$hidden_layer_size)
                          self$Who <- matrix(runif((self$hidden_layer_size)), ncol = 1)
                          self$a <- matrix(runif(self$hidden_layer_size*self$dim), ncol = self$dim)
                        }
                        , forward = function(x){
                          h <- as.matrix(x)%*%self$Wih
                          self$a <- sigmoid(h)
                          y <- sigmoid(self$a %*% self$Who) #Output of the network
                          return(y)
                        }
                        , backward = function(t,y,X){
                          
                          # Compute the error in the output layer
                          layer2_error <-  t-y
                          layer2_delta <- (layer2_error)*(y*(1-y)) 

                          #Compute the error in the input layer
                          layer1_error <- layer2_delta %*% t(self$Who)
                          layer1_delta <- layer1_error*self$a*(1-self$a)
                          
                          # Adjustments of the weights
                          layer1_adjustment <- t(X) %*% layer1_delta
                          layer2_adjustment <- t(self$a) %*% layer2_delta
                          
                          self$Wih <- self$Wih+self$learning_rate*layer1_adjustment
                          self$Who <- self$Who+self$learning_rate*layer2_adjustment

                        }
                        
                        , train = function(X,t){
                          n_examples <- nrow(X)
                          for(iter in 1:self$n_iter){
                            preds <- self$forward(X)
                            self$backward(t,preds, X)
                            if(iter %% 1000 == 0){
                              cat("Iteration: ", iter,"\n")
                            }
                            
                          }
                        }
                        , predict = function(X){
                          preds <- self$forward(X)
                          return(preds)
                        }
                      )
)

###########################################################################
#################
#################   TEST
#################
###########################################################################
x1 <- c(0,0,1,1)
x2 <- c(0,1,0,1)
t <- c(0,1,1,1)
X <- as.matrix(data.frame(x1=x1, x2=x2))

clf <- MLP$new(n_iter=5000,dim=ncol(X), hidden_layer_size=4)
clf$train(X,t)
clf$predict(X)


#That's all very nice, but what abou the XOR? Not linearly separable
xor <- data.frame(x1=c(0,0,1,1), x2=c(0,1,0,1), t = c(0,1,1,0))
clf$train(xor[,1:2],xor[,3])
clf$predict(xor[,1:2])

library(ggplot2)
grid_size <- 1e2
grid <- data.frame(V1=0,V2=0)
base <- seq(0,1,1/grid_size)


for(j in 1:grid_size){
    V1 <- rep(base[j],grid_size+1)
    V2 <- base
    tmp <- data.frame(V1=V1,V2=V2)
    grid <- rbind(tmp,grid)
}

grid
grid$z <- with(grid,clf$predict(cbind(V1,V2)))
ggplot(grid,aes(x=V1,y=V2))+geom_tile(aes(fill=z))+theme_bw()

