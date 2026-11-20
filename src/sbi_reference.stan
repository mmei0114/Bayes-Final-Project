data {
  int<lower=1> N;
  int<lower=1> T;
  int<lower=2> K;
  int<lower=1, upper=3> family;
  vector[T] t;
  array[N, T] int<lower=0> y;
}
transformed data {
  vector[N] y_sum = rep_vector(0, N);
  vector[N] yt_sum = rep_vector(0, N);
  if (family == 2) {
    for (i in 1:N) {
      for (j in 1:T) {
        y_sum[i] += y[i, j];
        yt_sum[i] += y[i, j] * t[j];
      }
    }
  }
}
parameters {
  ordered[K] intercept;
  vector[K] slope;
  vector<lower=log(0.005), upper=log(5)>[K] log_variance;
  simplex[K] pi;
  vector[N] z;
}
transformed parameters {
  vector[K] variance = exp(log_variance);
}
model {
  vector[K] sd = sqrt(variance);
  vector[K] log_pi = log(pi);
  matrix[T, K] time_effect;
  vector[K] poisson_exposure = rep_vector(0, K);
  intercept ~ normal(0, 1.5);
  slope ~ normal(0, 1.5);
  log_variance ~ normal(log(0.5), 0.7);
  pi ~ dirichlet(rep_vector(2, K));
  z ~ std_normal();

  for (k in 1:K) {
    time_effect[, k] = t * slope[k];
    if (family == 2) {
      poisson_exposure[k] = sum(exp(time_effect[, k]));
    }
  }
  for (i in 1:N) {
    vector[K] lp;
    for (k in 1:K) {
      real level = intercept[k] + sd[k] * z[i];
      if (family == 1) {
        lp[k] = log_pi[k] + bernoulli_logit_lpmf(y[i] | level + time_effect[, k]);
      } else if (family == 2) {
        // The data-only sum of log factorials cancels in the target.
        lp[k] = log_pi[k] + y_sum[i] * level + yt_sum[i] * slope[k]
                - exp(level) * poisson_exposure[k];
      } else {
        lp[k] = log_pi[k] + neg_binomial_2_log_lpmf(y[i] | level + time_effect[, k], 5.0);
      }
    }
    // Sum U_i out before sampling. Conditional on U_i=k, b_i=sqrt(D_k)*z_i.
    target += log_sum_exp(lp);
  }
}
