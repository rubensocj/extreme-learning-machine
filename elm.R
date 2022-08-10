#' @Title
#' Extreme learning machine (ELM)
#' 
#' @description 
#' Implementação de redes neurais de camada oculta única usando o
#' algoritmo de treinamento extreme learning machine (ELM).
#'
#' @param Y matrix; variável dependente.
#' @param X matrix; covariáveis.
#' @param h integer; tamanho da camada oculta.
#' @param act.fun function; função de ativação.
#' @param dist function; distribuição de probabilidades para os pesos.
#'
#' @author Rubens Oliveira da Cunha Júnior (cunhajunior.rubens@gmail.com).
#' 
#' @return list;
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
  
  # fitted
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

softsign <- function(x) x / (abs(x) + 1)

relu <- function(x) ifelse(x > 0, x, 0)

identity <- function(x) x