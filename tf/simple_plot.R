library(ggplot2)

args<-commandArgs(TRUE)
PLOT_TRAIN<-FALSE

d<-args[1]
noise<-args[2]

fname_base<-paste0('output/simple/fft1_d', d, '_noise', noise, '_bn20_nb50000_')
print(fname_base)

# --- vali --- #
fname<-paste0(fname_base, 'vali.txt')
data<-read.table(fname, header=T)
data['rep']<-NULL
data['method']<-NULL

ggplot(data, aes(x=training_examples, y=loss, colour=experiment, group=experiment, fill=experiment)) +  ggtitle(paste0("validation set loss (d=", d, ")")) + xlab("# training examples seen") + theme_bw() + stat_summary(fun.data = "mean_se", geom = "smooth")  + theme(legend.position="bottom") + ylim(0, 10)
ggsave(gsub(".txt", ".png", fname))

# --- train --- # (copy pasta)
if (PLOT_TRAIN){
    fname<-paste0(fname_base, 'train.txt')
    data<-read.table(fname, header=T)
    data['rep']<-NULL

    ggplot(data, aes(x=training_examples, y=loss, colour=experiment, group=experiment, fill=experiment)) +  ggtitle(paste0("training set loss (d=", d, ")")) + xlab("# training examples seen") + theme_bw() + stat_summary(fun.data = "mean_se", geom = "smooth")  + theme(legend.position="bottom")
    ggsave(gsub(".txt", ".png", fname))
}

# --- print summary statistics about test --- #
fname<-paste0(fname_base, 'test.txt')
data<-read.table(fname, header=T)

means<-aggregate(data$loss, by=list(data$experiment), FUN=mean)
names(means)<-c("experiment", "mean")
sems<-aggregate(data$loss, by=list(data$experiment), FUN=function(x) sd(x)/sqrt(length(x)))
names(sems)<-c("experiment", "standard error")

test_results<-merge(means, sems)
print(test_results)
