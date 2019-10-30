library(dplyr)
library(tidyr)
library(tidyverse)
library(data.table)

setwd('D:/xplore_idm/')

dhs = fread('data/dhs_gps.csv')
nightlight = fread('data/nightlight_values.csv')


# Process nightlight values
newCols = as.character(lapply(
  1:2025,
  function(x) { return(paste0('V', as.character(x))) }))
nightlight = nightlight %>% 
  mutate(first=gsub('\\[|\\]', '', first)) %>%
  separate('first', into=newCols, sep='[,]', convert=TRUE) %>%
  select(2:2027) %>%
  arrange(cluster_id)
nightlight[1:5, 1:5]
fwrite(nightlight, 'data/nightlight_values_processed.csv')

values = data.matrix(nightlight[2:2026])
values = matrix(values, 1800225, 1)

hist(pmin(values, 1), breaks=100, xlab='Intensity', main='Night Light Intensity, capped at 1')
hist(log(values+.1), breaks=100, xlab='Intensity', main='Log(intensity + 0.1)')

lvalues = log(data.matrix(nightlight[2:2026]+.1))
buckets = matrix(NA, 889, 28)
for (i in 1:889){
  buckets[i, ] = hist(lvalues[i, ], breaks=seq(-5, 9, by=0.5), plot=FALSE)$counts
}
buckets = cbind(buckets, rowMeans(lvalues))
buckets = as.data.table(buckets)
buckets = bind_cols(select(nightlight, cluster_id), buckets)
fwrite(buckets, 'data/nightlight_buckets.csv')
