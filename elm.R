#' @Title
#' Extreme learning machine (ELM)
#' 
#' @description 
#' Redes neural de camada oculta única usando o
#' algoritmo de treinamento extreme learning machine (ELM).
#'
#' @param Y matrix; resposta.
#' @param X matrix; covariáveis.
#' @param h integer; tamanho da camada oculta.
#' @param act.fun function; função de ativação.
#'
#' @author Rubens Oliveira da Cunha Junior (cunhajunior.rubens@gmail.com).
#' 
#' @return list;
elm <- function(Y, X, h, act.fun) {
  
  X <- as.matrix(X)
  Y <- as.matrix(Y)
  
  n.i <- ncol(X) # input nodes
  n.o <- ncol(Y) # output nodes
  
  # randomly initializes weights and bias
  W <- matrix(data = runif(n = (n.i + 1) * h, min = -1, max = 1),
              nrow = n.i + 1,
              ncol = h * n.o)
  
  # compute hidden layer output matrix: H
  H <- act.fun(cbind(1, X) %*% W)
  
  # compute H_ (invert H)
  H_ <- MASS::ginv(H)
  
  # compute beta
  beta <- H_ %*% Y
  
  # fitted
  pred <- H %*% beta
  
  return(list(fitted = pred, weights = W, act.fun = act.fun,
              beta = beta, H = H, X = X, Y = Y, h = h))
}

predict.elm <- function(model, new.data) {
  # compute H
  H <- model$act.fun(cbind(1, new.data) %*% model$weights)
  
  # make predictions
  pred <- H %*% model$beta
  
  return(pred)
}

sigmoid <- function(x) 1 / (1 + exp(-x))

tanh <- function(x) sinh(x) / cosh(x)

radial <- function(x) exp(-x^2)

softplus <- function(x) log(1 + exp(x))

softsign <- function(x) x / (abs(x) + 1)

relu <- function(x) ifelse(x > 0, x, 0)

identity <- function(x) x