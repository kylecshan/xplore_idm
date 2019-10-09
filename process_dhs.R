library(foreign)
library(dplyr)

setwd('D:/xplore_idm/')

## Load DHS children's vaccination data and take out the columns of interest
# vacc_data <- read.dta('data/NGKR6ADT/NGKR6AFL.DTA', convert.factors=FALSE)
# vacc_cols <- c('v001', 'h2','h9','h3','h5','h7','h0','h4','h6','h8', 'h1', 'h10')
# vacc_data <- vacc_data[, vacc_cols]
# new_colnames <- c('cluster_id', 'bcg', 'measles', 'dpt1', 'dpt2', 'dpt3', 
#                'polio0', 'polio1', 'polio2', 'polio3', 
#                'health_card', 'any_vacc')
# colnames(vacc_data) <- new_colnames
# write.table(vacc_data, 'data/dhs_raw.csv')

## Process survey response codes and summarize by region
vacc_data <- read.table('data/dhs_raw.csv')
for (c in colnames(vacc_data)[2:12]) {
  # We're counting 8 (don't know) and 9 (missing) as NA
  vacc_data[, c] <- recode(vacc_data[, c], `0`=0, `1`=1, `2`=1, `3`=1, .default=as.double(NA))
}
vacc_summary <- vacc_data %>% group_by(cluster_id) %>% summarise_all(funs(mean(., na.rm=TRUE)))

## Read location data
gps_data <- read.dbf('data/NGGE6AFL/NGGE6AFL.DBF')
gps_cols <- c('DHSCLUST', 'LATNUM', 'LONGNUM')
gps_data <- gps_data[, gps_cols]
colnames(gps_data) <- c('cluster_id', 'latitude', 'longitude')

## Remove locations with lat/long = 0/0
gps_data = gps_data[gps_data$latitude != 0 & gps_data$longitude != 0, ]
gps_data <- tbl_df(gps_data)

## Combine with vaccination data
data <- vacc_summary %>% inner_join(gps_data, by='cluster_id')
write.table(data, 'data/dhs_gps.csv', sep=',', row.names=FALSE)

## Export just the cluster locations
write.table(as.data.frame(gps_data[c('longitude','latitude','cluster_id')]), 
            'data/gps_raw.csv', sep=',', row.names=FALSE)

write.table(as.data.frame(gps_data[1:5, c('longitude','latitude','cluster_id')]), 
            'data/gps_raw_small.csv', sep=',', row.names=FALSE)
