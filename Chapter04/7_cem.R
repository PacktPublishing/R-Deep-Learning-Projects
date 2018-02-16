library(MASS)

# the function we want to optimize
f <-  function(theta){
  reward = -sum((solution - theta)**2)
  return(reward)
}

  
  
  solution <- c(0.5, 0.1, -0.3)
  dim_theta <-  length(solution)
  theta_mean <-  matrix(0,dim_theta,1)
  theta_std <-  diag(dim_theta)
  batch_size <-  25 # number of samples per batch
  elite_frac <-  0.2 # fraction of samples used as elite set
  
cem <- function(f, n_iter, theta_mean, theta_std, batch_size=25, elite_frac=0.2){
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

cem(f,300, theta_mean, theta_std)  
    