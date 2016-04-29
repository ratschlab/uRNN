library(ggplot2)

args<-commandArgs(TRUE)

d<-args[1]

fname<-paste0('output/simple_d', d, '_bn100_nb10000_vali.txt')

data<-read.table(fname)
names(data)<-c('parametrisation', 'training_examples', 'loss')

ggplot(data, aes(x=training_examples, y=loss, colour=parametrisation)) + geom_point() + ggtitle(paste0("validation set loss (d=", d, ")")) + xlab("# training examples seen") + theme_bw()
ggsave(gsub(".txt", ".png", fname))
