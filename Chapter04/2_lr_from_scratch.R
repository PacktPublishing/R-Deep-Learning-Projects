library(R6)

logit <- function(x){
  1/(1+exp(-x))
}

LR <- R6Class("LR", 
                      public = list(
                        dim = NULL,
                        n_iter = NULL,
                        learning_rate = NULL,
                        w = NULL,
                        initialize = function(learning_rate = 0.25, n_iter=100, dim=2){
                          self$n_iter <- n_iter
                          self$learning_rate <- learning_rate
                          self$dim <- dim
                          self$w <- matrix(runif(self$dim+1), ncol = self$dim+1)
                        }
                        , forward = function(x){
                          dot_product <- sum(x*self$w)
                          y <- logit(dot_product)
                          return(y)
                        }
                        , backward = function(t,y,x){
                          for(j in 1:ncol(x)){
                            self$w[j] <- self$w[j]+self$learning_rate*(t-y)*x[j]*logit(x[j])*(1-logit(x[j]))
                          }
                          
                        }
                        , train = function(X,t){
                          X <- cbind(-1,X) #add bias term
                          n_examples <- nrow(X)
                          
                          for(iter in 1:self$n_iter){
                            for(i in 1:nrow(X)){
                              y_i <- self$forward(X[i,])
                              self$backward(t[i],y_i, X[i,])
                            }
                            if(iter %% 20 == 0){
                              cat("Iteration: ", iter)
                              print("Weights: ")
                              print(unlist(self$w))  
                            }
                            
                          }
                        }
                        , predict = function(X){
                          X <- cbind(-1,X) #add bias
                          preds <- c()
                          for(i in 1:nrow(X)){
                            preds[i] <- self$forward(X[i,])
                          }
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
X <- data.frame(x1=x1, x2=x2)

lr <- LR$new(n_iter=100, dim=ncol(X))
lr
lr$train(X,t)
lr$w
lr$predict(X)


df <- as.data.frame(X)
df$t <- as.factor(t)


# Get the line
w0 <- as.numeric(lr$w[1])
w1 <- as.numeric(lr$w[2])
w2 <- as.numeric(lr$w[3])

x1_vals <- seq(-0.15,1,0.1)
x2_vals <- (w0-w1*x1_vals)/w2
boundary <- data.frame(x1_vals=x1_vals, x2_vals=x2_vals)


## Plot decision boundary
library(ggplot2)
ggplot()+geom_point(data=df, aes(x=x1,y=x2, color=t, size=2))+geom_line(data=boundary, aes(x=x1_vals, y=x2_vals, size=1))+theme_bw()



#That's all very nice, but what abou the XOR? Not linearly separable

xor <- data.frame(x1=c(0,0,1,1), x2=c(0,1,0,1), t = c(0,1,1,0))
lr$train(xor[,1:2],xor[,3])
lr$predict(xor[,1:2])

xor$t <- as.factor(xor$t)
ggplot()+geom_point(data=xor, aes(x=x1,y=x2, color=t, size=2))+theme_bw()
