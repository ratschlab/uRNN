#!/usr/bin/env R
# visualisation for adding task!
library(ggplot2)

base_dir<-"/home/hyland/git/complex_RNN/tf/output/adding"

args<-commandArgs(TRUE)
T_val<-args[1]

# --- constants --- #
batch_size<-20
batch_skip<-150

# --- IRNN --- #
IRNN_fname<-ifelse(T_val==750, "IRNN_T750_n80.vali.txt", paste0("lr1e-4_IRNN_T", T_val, "_n80.vali.txt"))
IRNN_trace<-read.table(paste0(base_dir, "/T", T_val, "/", IRNN_fname), header=T)
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

# --- complexRNN --- #
complex_RNN_trace<-read.table(paste0("/home/hyland/git/complex_RNN/addingT", T_val, "_tf.vali.txt"), header=TRUE)
batch_skip <- 50            # NOTE DIFFERENT
num_updates <- batch_skip * seq(nrow(complex_RNN_trace))
num_examples<- batch_size * num_updates
cost <- complex_RNN_trace$vali_cost
which <- rep("complex_RNN", nrow(complex_RNN_trace))
dtemp<-data.frame(num_updates, num_examples, cost, which)

da<-rbind(da, dtemp)

# --- NOW FOR PLOT --- #
ggplot(da, aes(x=num_updates, y=cost, group=which, colour=which)) + geom_point(cex=0.3) +  geom_line(alpha=0.2) + coord_cartesian(ylim=c(0, 0.25)) + ggtitle(paste0("adding, T = ", T_val))
ggsave(paste0(base_dir, "/adding_T", T_val, ".png"), width=4.5, height=3)
