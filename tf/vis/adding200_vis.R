#!/usr/bin/env R
# visualisation for adding task!
library(ggplot2)

base_dir<-"/home/hyland/git/complex_RNN/tf/output/adding"

# --- IRNN --- #
#IRNN_trace<-read.table(paste0(base_dir, "/T200/lr1e-5_IRNN_T200_n80.vali.txt"), header=TRUE)
IRNN_trace<-read.table(paste0(base_dir, "/lr1e-4_IRNN_T200_n80.vali.txt"), header=TRUE)
batch_size<-20
batch_skip<-150
num_updates <- batch_skip * seq(nrow(IRNN_trace))
num_examples<- batch_size * num_updates
cost <- IRNN_trace$vali_cost
which <- rep("IRNN", nrow(IRNN_trace))

da<-data.frame(num_updates, num_examples, cost, which)

# --- LSTM --- #
LSTM_trace<-read.table(paste0(base_dir, "/T200/LSTM_T200_n40.vali.txt"), header=TRUE)
batch_size<-20
batch_skip<-150
num_updates <- batch_skip * seq(nrow(LSTM_trace))
num_examples<- batch_size * num_updates
cost <- LSTM_trace$vali_cost
which <- rep("LSTM", nrow(LSTM_trace))
dtemp <- data.frame(num_updates, num_examples, cost, which)

da<-rbind(da, dtemp)

# --- tanhRNN --- #
tanhRNN_trace<-read.table(paste0(base_dir, "/T200/tanhRNN_T200_n80.vali.txt"), header=TRUE)
batch_size<-20
batch_skip<-150
num_updates <- batch_skip * seq(nrow(tanhRNN_trace))
num_examples<- batch_size * num_updates
cost <- tanhRNN_trace$vali_cost
which <- rep("tanhRNN", nrow(tanhRNN_trace))
dtemp<-data.frame(num_updates, num_examples, cost, which)

da<-rbind(da, dtemp)

# --- NOW FOR PLOT --- #
ggplot(da, aes(x=num_updates, y=cost, group=which, colour=which)) + geom_point(cex=0.3) + geom_line(alpha=0.2) + coord_cartesian(ylim=c(0, 0.25)) + ggtitle("adding T = 200")
ggsave(paste0(base_dir, "/adding_T200.png"), width=4.5, height=3)
