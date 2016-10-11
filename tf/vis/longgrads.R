#!/usr/bin/env R

library(ggplot2)
base_dir <- "/Users/stephanie/PhD/git/complex_RNN/tf/output/memory/"

da<-data.frame()
fname<-'gradtestlong_tanh_uRNN_T100_n30.hidden_gradients.txt'
da<-read.table(paste0(base_dir, fname), header=T)

# extract just one k value
k_pick<-15
da<-subset(da, k==k_pick)

# because in this version, I forgot to include the epoch, we have to deduce it
batches_per_epoch = len(unique(da$batch))
epoch<-rep(seq(batches_per_epoch), nrow(da)/batches_per_epoch)
da<-data.frame(epoch, da)

# --- now for plot --- #
#ggplot(da, aes(x=k, y=norm, group=which, colour=which)) + geom_point(cex=0.3) + geom_line(alpha=0.2) + facet_grid(batch~.) + ggtitle("cost gradient wrt hidden state h_k") + xlab("k") + ylab("|dC/dh_k|") + scale_y_log10()

#ggsave(paste0(base_dir, "internal_gradients_all.png"), width=4.5, height=3)
