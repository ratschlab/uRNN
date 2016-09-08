#!/usr/bin/env R
# visualisation for memory task!
library(ggplot2)

base_dir<-"/home/hyland/git/complex_RNN/tf/output/memory"

args<-commandArgs(TRUE)
T_val<-args[1]

# --- constants --- #
batch_size<-20
batch_skip<-150

# --- IRNN --- #
IRNN_trace<-read.table(paste0(base_dir, "/T", T_val, "/lr1e-4_IRNN_T", T_val, "_n80.vali.txt"), header=TRUE)
num_updates <- batch_skip * seq(nrow(IRNN_trace))
num_examples<- batch_size * num_updates
cost <- IRNN_trace$vali_cost
which <- rep("IRNN", nrow(IRNN_trace))

da<-data.frame(num_updates, num_examples, cost, which)

# --- LSTM --- #
LSTM_trace<-read.table(paste0(base_dir, "/T", T_val, "/LSTM_T", T_val, "_n40.vali.txt"), header=TRUE)
num_updates <- batch_skip * seq(nrow(LSTM_trace))
num_examples<- batch_size * num_updates
cost <- LSTM_trace$vali_cost
which <- rep("LSTM", nrow(LSTM_trace))
dtemp <- data.frame(num_updates, num_examples, cost, which)

da<-rbind(da, dtemp)

# --- tanhRNN --- #
tanhRNN_trace<-read.table(paste0(base_dir, "/T", T_val, "/tanhRNN_T", T_val, "_n80.vali.txt"), header=TRUE)
num_updates <- batch_skip * seq(nrow(tanhRNN_trace))
num_examples<- batch_size * num_updates
cost <- tanhRNN_trace$vali_cost
which <- rep("tanhRNN", nrow(tanhRNN_trace))
dtemp<-data.frame(num_updates, num_examples, cost, which)

da<-rbind(da, dtemp)

#--- NOW FOR PLOT --- #
xmax <- ifelse(T_val %in% c(100, 200), 50000, 100000)
ggplot(da, aes(x=num_updates, y=cost, group=which, colour=which)) + geom_point(cex=0.3) +  geom_line(alpha=0.2) + coord_cartesian(xlim=c(0, xmax), ylim=c(0, 0.3)) + ggtitle(paste0("memory T=", T_val))
ggsave(paste0(base_dir, "/memory_T", T_val, ".png"), width=4.5, height=3)
