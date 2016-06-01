library(ggplot2)

PLOT_TRAIN<-FALSE

d<-6
#identifier<-args[2]
noise<-0.01

fname_base<-paste0('output/simple/nips/random_projections_d', d, '_noise', noise, '_bn20_nb50000_')
print(fname_base)

# --- vali --- #
fname<-paste0(fname_base, 'vali.txt')
data<-read.table(fname, header=T)
print(levels(factor(data$rep)))
data['method']<-NULL
data['experiment']<-factor(data$experiment, labels=c('full', 'J=16', 'J=25', 'J=36', 'J=4', 'J=9'))
print(levels(data$experiment))
data$experiment<-factor(data$experiment, levels=c('J=4', 'J=9', 'J=16', 'J=25', 'J=36', 'full'))
print(levels(data$experiment))

# omg using geom_ribbon with different x values is... hard, I need to do smoothing
# ... but have no time, so picking a random rep (they all seem roughly the same)
ggplot(subset(data, rep==2), aes(x=t, y=loss, group=experiment, color=experiment)) + geom_point(cex=0.9) +  xlab("training time (s)") + ylab("validation set loss") + theme_bw()
ggsave(gsub(".txt", "_time.png", fname), width=5.5, height=4)
ggsave(gsub(".txt", "_time.pdf", fname), width=5.5, height=4)

data['rep']<-NULL

ggplot(data, aes(x=training_examples/1e6, y=loss, colour=experiment, group=experiment, fill=experiment)) +  xlab("training examples seen (millions)") + ylab("validation set loss") + theme_bw() + stat_summary(fun.data = "mean_se", geom = "smooth")  + ylim(0, 5)
ggsave(gsub(".txt", ".png", fname), width=5.5, height=4)
ggsave(gsub(".txt", ".pdf", fname), width=5.5, height=4)


# --- print summary statistics about test --- #
#fname<-paste0(fname_base, 'test.txt')
#dtest<-read.table(fname, header=T)

#means<-aggregate(dtest$loss, by=list(dtest$experiment), FUN=mean)
#names(means)<-c("experiment", "mean")
#sems<-aggregate(dtest$loss, by=list(dtest$experiment), FUN=function(x) sd(x)/sqrt(length(x)))
#names(sems)<-c("experiment", "standard error")

#test_results<-merge(means, sems)
#print(test_results)
