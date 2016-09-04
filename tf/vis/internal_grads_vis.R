#!/usr/bin/env R

library(ggplot2)

#args<-commandArgs(TRUE)
#args<-"gradnorms_test_tanhRNN_T100_n3.hidden_gradients.txt"
#$args<-"gradnorms_test_LSTM_T100_n3.hidden_gradients.txt"

base_dir <- "/home/hyland/git/complex_RNN/tf/output/adding/"

da<-data.frame()

# --- IRNN --- #
#IRNN_grads<-read.table(paste0(base_dir, "gradnorms_test_IRNN_T100_n3.hidden_gradients.txt"), header=TRUE)
#which<-rep("IRNN", nrow(IRNN_grads))
#da<-rbind(data.frame(IRNN_grads, which))

# --- LSTM --- #
LSTM_grads<-read.table(paste0(base_dir, "gradnorms_test_LSTM_T100_n3.hidden_gradients.txt"), header=TRUE)
which<-rep("LSTM", nrow(LSTM_grads))
da<-rbind(da, data.frame(LSTM_grads, which))

# --- tanhRNN --- #
tanhRNN_grads<-read.table(paste0(base_dir, "gradnorms_test_tanhRNN_T100_n3.hidden_gradients.txt"), header=TRUE)
which<-rep("tanhRNN", nrow(tanhRNN_grads))
da<-rbind(da, data.frame(tanhRNN_grads, which))

# --- complex_RNN --- #
#complex_RNN_grads<-read.table(paste0(base_dir, "gradnorms_test_complex_RNN_T100_n3.hidden_gradients.txt"), header=TRUE)
#which<-rep("complex_RNN", nrow(complex_RNN_grads))
#da<-rbind(data.frame(complex_RNN_grads, which))

# --- uRNN --- #
uRNN_grads<-read.table(paste0(base_dir, "gradnorms_test_uRNN_T100_n3.hidden_gradients.txt"), header=TRUE)
which<-rep("uRNN", nrow(uRNN_grads))
da<-rbind(da, data.frame(uRNN_grads, which))

# --- now for plot --- #
ggplot(da, aes(x=k, y=norm, group=which, colour=which)) + geom_point(cex=0.3) + geom_line(alpha=0.2) + facet_grid(batch~.) + ggtitle("norm of gradient of cost with respect to hidden state h_k") + xlab("k") + ylab("|dC/dh_k|") + scale_y_log10()

ggsave(paste0(base_dir, "internal_gradients_all.png"), width=4.5, height=3)
