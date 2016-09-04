#!/usr/bin/env bash
# visualse the norm of the difference between the final vector...
library(ggplot2)

#args<-commandArgs(TRUE)
base_dir <- "/home/hyland/git/complex_RNN/tf/output/adding/"
#args<-"gradnorms_test_tanhRNN_T100_n3.hidden_states.txt"
#args<-"gradnorms_test_LSTM_T100_n3.hidden_states.txt"
args<-"gradnorms_test_complex_RNN_T100_n3.hidden_states.txt"

states_path <- paste0(base_dir, args[1])
da<-read.table(states_path, header=T)
da$batch <- factor(da$batch)

# now, for each batch... (there will be two)
# this is... very gross, sorry world
#`da_plot <- data.frame()
#`for (batch in levels(da$batch)){
 #`   da_sub <- subset(da, batch==batch)
  #`  k_max <- max(da_sub$k)
   #` final_state = subset(da_sub, k==k_max)[1, 3:ncol(da_sub)]
 #`   for (ki in seq(k_max-1)){
 #`       state_k <- subset(da_sub, k==ki)[1,3:ncol(da_sub)]
#`        stopifnot(nrow(subset(da_sub, k==ki)) == 1)
#`        norm_diff <-sqrt(sum((final_state - state_k)^2))
#`        da_plot <- rbind(da_plot, c(batch, ki, norm_diff))
#`    }
#`}

#`names(da_plot) <- c("batch", "k", "norm")

ggplot(da, aes(x=k, y=value, colour=what, group=what)) + geom_point(cex=0.3) + geom_line(alpha=0.2) + facet_grid(batch~.) + ggtitle("norm of difference between internal state h_k and final state") + xlab("k") + ylab("|h_k|")
ggsave(gsub(".txt", ".png", states_path), width=4.5, height=3)

