library(dplyr)
library(tidyr)

setwd('D:/xplore_idm/')

dhs = read.table('data/dhs_gps.csv', sep=',', header=TRUE)
nightlight = read.table('data/nightlight_buckets.csv', sep=',', header=TRUE, as.is=TRUE)

# Process nightlight intensity bucket data
thresholds = as.character(lapply(
  c(0, 0.01, 0.0178, 0.0316, 0.0562, 
  0.1, 0.178, 0.316, 0.562, 
  1, 1.78, 3.16, 5.62,
  10, 17.8, 31.6, 1000000),
  function(x) { return(as.character(x)) }))
vacc_cols = c('bcg', 'measles', 'dpt1', 'dpt2', 'dpt3', 
              'polio0', 'polio1', 'polio2', 'polio3',
              'health_card', 'any_vacc')
nightlight$values <- lapply(nightlight$values, function(x) { return(substring(x, 2, nchar(x)-1))})
nightlight <- separate(nightlight, col='values', into=thresholds, sep=',', convert=TRUE)
# write.table(nightlight[c('cluster_id', thresholds)], 'data/nightlight_buckets_parsed.csv', sep=',', row.names=FALSE)

data = inner_join(dhs, nightlight, by='cluster_id')
X = as.matrix(data[thresholds])
Y = as.matrix(data[vacc_cols])

r2s = data.frame(matrix(NA, 1, ncol(Y)))
colnames(r2s) <- vacc_cols
for (j in 1:ncol(Y)) {
  model <- lm(Y[,j] ~ X)
  r2s[j] <- summary(model)$r.squared
}
