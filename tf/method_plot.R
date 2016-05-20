#!/usr/bin/env R
library(ggplot2)

n_pick<-8

dbox<-read.table('output/simple/nips/boxplot_compile.txt', header=T)
#dbox<-read.table('output/simple/nips/d8_noise0.01_bn20_nb50000_test.txt', header=T)
dd<-subset(dbox, experiment %in% c('complex_RNN', 'general_unitary'))
dd$experiment<-factor(dd$experiment, labels=c('comp', 'u(n)'))
print(levels(dd$experiment))
dd$method<-factor(dd$method, labels=c('composition', 'Lie algebra', 'QR'))

p<-ggplot(subset(dd, n==n_pick), aes(experiment, loss)) + theme_bw() + ylab("test set loss") + ggtitle(paste0('n = ', n_pick)) + xlab("approach")
p+geom_boxplot(aes(fill=method, color=method))

ggsave(paste0('output/simple/nips/boxplot', n_pick, '.png'), width=5.5, height=4)
ggsave(paste0('output/simple/nips/boxplot', n_pick, '.pdf'), width=5.5, height=4)

