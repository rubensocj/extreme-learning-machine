#' @Title
#' Ensemble extreme learning machine (E-ELM)
#' 
#' @description 
#' Ensemble baseado na média de redes neurais de camada oculta única usando
#' o algoritmo de treinamento extreme learning machine (ELM).
#'
#' @param Y matrix; resposta.
#' @param X matrix; covariáveis.
#' @param P integer; número de modelos.
#' @param h integer; tamanho da camada oculta.
#' @param act.fun function; função de ativação.
#'
#' @author Rubens Oliveira da Cunha Júnior (cunhajunior.rubens@gmail.com).
#' 
#' @return list;
eelm <- function(Y, X, P, h, act.fun) {
  
  # P ELM models
  models <- replicate(P, elm(Y = Y, X = X, h = h, act.fun = act.fun), FALSE)
  
  # Fitted values
  f <- lapply(models, "[", 1)
  df <- cbind.data.frame(f)
  
  # Ensemble of fitted values
  e.f <- apply(df, 1, mean)
  
  return(list(models = models, fitted = e.f))
}

predict.eelm <- function(models, new.data) {

  # Predictions
  p <- lapply(models, predict.elm, new.data)
  df <- cbind.data.frame(p)
  
  # Ensemble of predictions
  e.p <- apply(df, 1, mean)
  
  return(list(ensemble = e.p, predictions = df))
}

sigmoid <- function(x) 1 / (1 + exp(-x))

tanh <- function(x) sinh(x) / cosh(x)

radial <- function(x) exp(-x^2)

softplus <- function(x) log(1 + exp(x))

softsign <- function(x) x / (abs(x) + 1)

relu <- function(x) ifelse(x > 0, x, 0)

identity <- function(x) x