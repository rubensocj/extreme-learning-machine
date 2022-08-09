#' Title
#'
#' @param Y matrix; variável dependente.
#' @param X matrix; covariáveis.
#' @param h integer; tamanho da camada oculta.
#' @param act.fun function; função de ativação.
#' @param dist function; distribuição de probabilidades para os pesos.
#'
#' @return list
#'
#' @examples
elm <- function(Y, X, h, act.fun = sigmoid, dist.fun = runif) {
  
  X <- as.matrix(X)
  Y <- as.matrix(Y)
  
  n.i <- ncol(as.matrix(X)) # input nodes
  n.o <- ncol(as.matrix(Y)) # output nodes
  
  # randomly initializes weights and bias
  W <- matrix(data = dist.fun(n = n.i * h), nrow = n.i, ncol = h)
  bias <- dist.fun(n = h)
  
  # compute H
  H <- act.fun(X %*% W + bias)
  
  # compute H_ (invert H)
  H_ <- MASS::ginv(H)
  
  # compute beta estimate
  beta <- H_ %*% Y
  
  # fitted (predictions)
  pred <- H %*% beta
  
  return(list(fitted = pred, weights = W, bias = bias, act.fun = act.fun,
              beta = beta))
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

