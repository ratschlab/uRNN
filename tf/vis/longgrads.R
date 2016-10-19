#!/usr/bin/env R

library(ggplot2)
base_dir <- "/Users/stephanie/PhD/git/complex_RNN/tf/output/memory/"

fname<-paste0(base_dir, 'gradtestlong_tanh_uRNN_T100_n30')

# --- grads --- #
grads<-paste0(fname, ".hidden_gradients.txt")
da_grads<-read.table(grads, header=T)

# extract just one k value
#k_pick<-100
# let's put a STAT on this
#da_grads<-subset(da_grads, k==k_pick)

# because in this version, I forgot to include the epoch, we have to deduce it
batches_per_epoch = length(unique(da_grads$batch))
# get epochs
epoch<-c()
epoch_counter<-0
prev_batch<-0
for (batch in da_grads$batch){
    if (batch < prev_batch){ epoch_counter <- epoch_counter + 1 } 
    prev_batch <- batch
    epoch<-c(epoch, epoch_counter)
}
updates<-(epoch*5000) + da_grads$batch
da_grads<-data.frame(updates, epoch, da_grads)

# --- now get the vali --- #
vali<-paste0(fname, ".vali.txt")
da_vali<-read.table(vali, header=T)
updates<-da_vali$batch + (5000*da_vali$epoch)
da_vali<-data.frame(updates, da_vali)
# delete the first row
da_vali <- da_vali[2:nrow(da_vali), ]

# combine them further
which<-c(rep("grad", nrow(da_grads)), rep("vali cost", nrow(da_vali)))
da_1<-data.frame(da_grads$updates, da_grads$norm)
names(da_1)<-c("update", "val")
# normalise (arbitrary)
da_1$val<-da_1$val/da_1$val[0.25*nrow(da_1)]

da_2<-data.frame(da_vali$updates, da_vali$vali_cost)
names(da_2)<-c("update", "val")
# normalise
da_2$val<-da_2$val/da_2$val[0.1*nrow(da_2)]

da<-rbind(da_1, da_2)
da<-data.frame(da, which)

# --- now for plot --- #
ggplot(da, aes(x=update, y=val, group=which, colour=which)) + stat_summary(geom="smooth", fun.data="mean_cl_boot") + xlab("number of training updates") + ylab("arbitrary scale") + theme_bw() 
ggsave(paste0(fname, ".longgrads.png"), width=4.5, height=3)

