#!/usr/bin/env R
# compare no-bias uRNN at n = 10, 20, 30
library(ggplot2)

base_dir<-"/home/hyland/git/complex_RNN/tf/output/adding/"

ten<-read.table(paste0(base_dir, 'relu_uRNN_T100_n10.vali.txt'), header=T)
twenty<-read.table(paste0(base_dir, 'relu_uRNN_T100_n20.vali.txt'), header=T)
thirty<-read.table(paste0(base_dir, 'relu_uRNN_T100_n30.vali.txt'), header=T)

batch_size <- 20
batch_skip <- 150

which <- rep(10, nrow(ten))
which <- c(which, rep(20, nrow(twenty)))
which <- c(which, rep(30, nrow(thirty)))
num_updates <- seq(nrow(ten))
num_updates <- c(num_updates, seq(nrow(twenty)))
num_updates <- c(num_updates, seq(nrow(thirty)))
num_updates <- batch_skip * num_updates

da<-rbind(ten, twenty)
da<-rbind(da, thirty)

da<-data.frame(da, which, num_updates)

ggplot(da, aes(x=num_updates, y=vali_cost)) + geom_point(cex=0.3) + geom_line(alpha=0.2)
ggsave(paste0(base_dir, "g1_statesize.png"), widht=4.5, height=3)
