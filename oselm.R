#' @Title
#' Online sequential extreme learning machine (ELM)
#' 
#' @description 
#' Implementação de redes neurais de camada oculta única usando a versão online
#' do algoritmo de treinamento extreme learning machine (ELM).
#'
#' @details 
#' Etapa 1: fase de inicialização.
#'
#' @param Y matrix; resposta.
#' @param X matrix; covariáveis.
#' @param h integer; tamanho da camada oculta.
#' @param act.fun function; função de ativação.
#' @param dist function; distribuição de probabilidades para os pesos.
#' @param ... parâmetros adicionais da função distribuição de probabilidades.
#'
#' @author Rubens Oliveira da Cunha Júnior (cunhajunior.rubens@gmail.com).
#' 
#' @return list;
#'
#' @examples
oselm.initialization <- function(Y, X, h, act.fun = tanh, dist.fun = rnorm, ...) {
  
  # small chunk of initial training data
  X <- as.matrix(X)
  Y <- as.matrix(Y)
  
  n.i <- ncol(X) # input nodes
  n.o <- ncol(Y) # output nodes
  
  # assign random input weights and bias
  W <- matrix(data = dist.fun(n = (n.i + 1) * h, ...),
              nrow = n.i + 1,
              ncol = h)
  
  # compute the initial hidden layer output matrix: H_0
  H_0 <- act.fun(cbind(1, X) %*% W)
  
  # compute initial beta
  # if t(H) %*% H is singular use smaller h or increase training data samples
  p_0 <- solve(t(H_0) %*% H_0)
  beta_0 <- p_0 %*% t(H_0) %*% Y
  
  # fitted
  pred <- H_0 %*% beta_0
  
  return(list(fitted = pred, weights = W, act.fun = act.fun,
              beta = beta_0, p = p_0))
}

#' @Title
#' Online sequential extreme learning machine (ELM)
#' 
#' @description 
#' Implementação de redes neurais de camada oculta única usando a versão online
#' do algoritmo de treinamento extreme learning machine (ELM).
#'
#' @details 
#' Etapa 2: fase de aprendizagem sequencial.
#'
#' @param model object; modelo construído usando oselm.initialization.
#' @param Y_k matrix; variável dependente.
#' @param X_k matrix; covariáveis.
#'
#' @author Rubens Oliveira da Cunha Júnior (cunhajunior.rubens@gmail.com).
#' 
#' @return list;
#'
#' @examples
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
  H_k <- act.fun(cbind(1, X_k) %*% W)
  
  # calculate the output weight beta_k+1
  I <- diag(nrow = nrow(H_k)) # identity matrix
  p_k <- p_0 - p_0 %*% t(H_k) %*% solve(I + H_k %*% p_0 %*% t(H_k)) %*% H_k %*% p_0
  beta_k <- beta_0 + p_k %*% t(H_k) %*% (Y_k - H_k %*% beta_0)
  
  # fitted
  pred <- H_k %*% beta_k
  
  return(list(fitted = pred, weights = W, act.fun = act.fun,
              beta = beta_k, p = p_k))
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