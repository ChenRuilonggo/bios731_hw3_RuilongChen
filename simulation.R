# ---------------------------------------------------------
# 1. Setup Data Generation
# ---------------------------------------------------------
set.seed(2026)
n <- 200
beta_true <- c(1, 0.3)
x_data <- rnorm(n)
X <- cbind(1, x_data) # Design matrix
eta <- X %*% beta_true
pi_true <- 1 / (1 + exp(-eta))
y <- rbinom(n, 1, pi_true)

# Starting values and stopping criterion
beta_start <- c(0, 0)
tol <- 1e-8
max_iter <- 500

# Function for Standard Error Calculation
get_se <- function(X, beta) {
  pi <- as.vector(1 / (1 + exp(-X %*% beta)))
  W <- diag(pi * (1 - pi))
  H <- -t(X) %*% W %*% X
  sqrt(diag(solve(-H)))
}

# ---------------------------------------------------------
# 2. Newton's Method (Problem 1.1)
# ---------------------------------------------------------
run_newton <- function(X, y, start, tol) {
  beta <- start
  iter <- 0
  start_time <- Sys.time()
  repeat {
    iter <- iter + 1
    pi <- as.vector(1 / (1 + exp(-X %*% beta)))
    grad <- t(X) %*% (y - pi)
    W <- diag(pi * (1 - pi))
    H <- -t(X) %*% W %*% X
    beta_new <- beta - solve(H) %*% grad
    if (sum(abs(beta_new - beta)) < tol || iter >= max_iter) break
    beta <- beta_new
  }
  end_time <- Sys.time()
  list(beta = as.vector(beta_new), iter = iter, time = end_time - start_time)
}

# ---------------------------------------------------------
# 3. MM Algorithm (Problem 1.2/2.3)
# ---------------------------------------------------------
# Note: For the MM update, we solve the equation from (C) for each j
run_mm <- function(X, y, start, tol) {
  beta <- start
  p_dim <- length(beta)
  iter <- 0
  start_time <- Sys.time()
  repeat {
    iter <- iter + 1
    beta_old_iter <- beta
    # Cycle through each coordinate j
    for (j in 1:p_dim) {
      # Use a univariate root finder for the transcendental equation in (C)
      f_obj <- function(bj) {
        eta_k <- X %*% beta
        term1 <- exp(eta_k) / (1 + exp(eta_k))
        sum_val <- -sum( (term1 * X[,j] * exp(-p_dim * X[,j] * beta[j])) * exp(p_dim * X[,j] * bj) ) + sum(y * X[,j])
        return(sum_val)
      }
      # Solve for new beta[j]
      beta[j] <- uniroot(f_obj, interval = c(-10, 10))$root
    }
    if (sum(abs(beta - beta_old_iter)) < tol || iter >= max_iter) break
  }
  end_time <- Sys.time()
  list(beta = beta, iter = iter, time = end_time - start_time)
}

# ---------------------------------------------------------
# 4. GLM and Optim
# ---------------------------------------------------------
# GLM
t1_glm <- Sys.time()
fit_glm <- glm(y ~ x_data, family = binomial)
t2_glm <- Sys.time()

# Optim (BFGS)
log_lik <- function(b, X, y) {
  pi <- 1 / (1 + exp(-X %*% b))
  -sum(y * log(pi) + (1 - y) * log(1 - pi)) # Negative log-likelihood
}
t1_opt <- Sys.time()
fit_opt <- optim(beta_start, log_lik, X = X, y = y, method = "BFGS", hessian = TRUE)
t2_opt <- Sys.time()

# ---------------------------------------------------------
# 5. Compile Results
# ---------------------------------------------------------
res_newton <- run_newton(X, y, beta_start, tol)
res_mm <- run_mm(X, y, beta_start, tol)

methods <- list(
  Newton = list(b = res_newton$beta, se = get_se(X, res_newton$beta), iter = res_newton$iter, time = res_newton$time),
  MM = list(b = res_mm$beta, se = get_se(X, res_mm$beta), iter = res_mm$iter, time = res_mm$time),
  GLM = list(b = coef(fit_glm), se = summary(fit_glm)$coefficients[,2], iter = fit_glm$iter, time = t2_glm - t1_glm),
  Optim = list(b = fit_opt$par, se = sqrt(diag(solve(fit_opt$hessian))), iter = fit_opt$counts[1], time = t2_opt - t1_opt)
)