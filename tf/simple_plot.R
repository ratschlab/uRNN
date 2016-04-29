library(ggplot2)

args<-commandArgs(TRUE)

#d<-args[1]
d<-3


fname<-'output/simple_lie_algebra_d1_bn100_nb10000_train.txt'
#fname<-paste0('output/simple_d', d, '_bn100_nb10000_vali.txt')

data<-read.table(fname, header=T)
data['rep']<-NULL

ggplot(data, aes(x=training_examples, y=loss, colour=experiment, group=experiment, fill=experiment)) +  ggtitle(paste0("validation set loss (d=", d, ")")) + xlab("# training examples seen") + theme_bw() + stat_summary(fun.data = "mean_se", geom = "smooth")  + theme(legend.position="bottom")
ggsave("test.png")
#ggsave(gsub(".txt", ".png", fname))
