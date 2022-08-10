# Online Sequential Extreme Learning Machine implementation
set.seed(123)

# Select parameters
sigmoid <- function(x) 1 / (1 + exp(-x))  # activation function: g
h <- 10                                   # hidden node number: Ñ
dados <- cars                             # data arrives sequentially
  
# Step 1: initialization phase
# H_0: initial hidden layer output matrix
# H_0 matrix is filled up for use in the learning phase
# Initialize the learning using a small chunk of initial training data
# the number of training data can be equal or close to Ñ (hidden node number)
# number of training data > number of hidden nodes
dados0 <- dados[1:20, ]

X <- as.matrix(dados0$speed)
Y <- as.matrix(dados0$dist)

n.i <- ncol(X) # input nodes
n.o <- ncol(Y) # output nodes

# assign random input weights and bias
W <- matrix(data = rnorm(n = n.i * h), nrow = n.i, ncol = h)
bias <- rnorm(n = h)

# compute H
H_0 <- softsign(X %*% W + bias)

# compute beta estimate
# if t(H) %*% H is singular use smaller network size (Ñ) or increase data number
p_0 <- solve(t(H_0) %*% H_0)
beta_0 <- p_0 %*% t(H_0) %*% Y

# set k = 0
k <- 0

# Step 2: sequential learning phase
# (k+1)th chunk: First chunk
dados1 <- dados[21:30, ]

X_k1 <- as.matrix(dados1$speed)

# calculate the partial hidden layer output matrix: H_k+1
H_k1 <- softsign(X_k1 %*% W + bias)

# set T_k+1
Y_k1 <- as.matrix(dados1$dist)

# calculate the output weight beta_k+1
I <- diag(nrow = nrow(H_k1)) # identity matrix
p_k1 <- p_0 - p_0 %*% t(H_k1) %*% solve(I + H_k1 %*% p_0 %*% t(H_k1)) %*% H_k1 %*% p_0
beta_k1 <- beta_0 + p_k1 %*% t(H_k1) %*% (Y_k1 - H_k1 %*% beta_0)

# set k = k + 1
# go to step 2

# Remarks:
# The chunk size does not need to be constant.

oselm.initialization <- function(Y, X, h, act.fun = tanh, dist.fun = rnorm) {
  # small chunk of initial training data
  X <- as.matrix(X)
  Y <- as.matrix(Y)
  
  n.i <- ncol(X) # input nodes
  n.o <- ncol(Y) # output nodes
  
  # assign random input weights and bias
  W <- matrix(data = dist.fun(n = n.i * h), nrow = n.i, ncol = h)
  bias <- dist.fun(n = h)
  
  # compute the initial hidden layer output matrix: H_0
  H_0 <- act.fun(X %*% W + bias)
  
  # compute beta_0
  # if t(H) %*% H is singular use smaller h or increase training data samples
  p_0 <- solve(t(H_0) %*% H_0)
  beta_0 <- p_0 %*% t(H_0) %*% Y
  
  # fitted
  pred <- H_0 %*% beta_0
  
  return(list(fitted = pred, weights = W, bias = bias, act.fun = act.fun,
              beta = beta_0, p = p_0))
}

oselm.learning <- function(model, X_k, Y_k) {
  W <- model$weights
  bias <- model$bias
  act.fun <- model$act.fun
  p_0 <- model$p
  beta_0 <- model$beta

  # new chunk of data  
  X_k <- as.matrix(X_k)
  Y_k <- as.matrix(Y_k)
  
  # compute the partial hidden layer output matrix: H_k+1
  H_k <- act.fun(X_k %*% W + bias)
  
  # calculate the output weight beta_k+1
  I <- diag(nrow = nrow(H_k)) # identity matrix
  p_k <- p_0 - p_0 %*% t(H_k) %*% solve(I + H_k %*% p_0 %*% t(H_k)) %*% H_k %*% p_0
  beta_k <- beta_0 + p_k %*% t(H_k) %*% (Y_k - H_k %*% beta_0)
  
  # fitted
  pred <- H_k %*% beta_k
  
  return(list(fitted = pred, weights = W, bias = bias, act.fun = act.fun,
              beta = beta_k, p = p_k))
}

predict.elm <- function(model, new.data) {
  # compute H
  H <- model$act.fun(new.data %*% model$weights + model$bias)
  
  # make predictions
  pred <- H %*% model$beta
  
  return(pred)
}

sigmoid <- function(x) 1 / (1 + exp(-x))

tanh <- function(x) sinh(x) / cosh(x)

radial <- function(x) exp(-x^2)

softplus <- function(x) log(1 + exp(x))

softmax <- function(x) exp(x) / sum(exp(x))

softsign <- function(x) x / (abs(x) + 1)

relu <- function(x) max(0, x)

identity <- function(x) x

data('mcycle', package = 'MASS')
times <- matrix(mcycle$times, ncol = 1)
accel <- mcycle$accel
my.elm <- elm(Y = accel, X = times, h = 100, act.fun = tanh, dist.fun = rnorm)
plot(times, accel, pch = 16, col = 'black')
points(x = times, y = my.elm$fitted, pch = 16, col = 'red')

set.seed(2)
x <- rnorm(100, 0, )
y1 <- x^3 - 2*x^2 - 3*x + 2
y2 <- x^2 + x - 1
elm2 <- elm(Y = cbind(y1, y2), X = x, h = 50)
elm2$beta

