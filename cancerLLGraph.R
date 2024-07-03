mp = read.csv("~/Documents/Northwestern/Research/Code/Model Performance - Cancer LL.csv",
              stringsAsFactors=F)
mp$Mean=as.numeric(sub(",", "", mp$Mean))
mp$Total.Topics=as.numeric(sub(",", "", mp$Total.Topics))

labels = c()
for (i in 1:nrow(mp)){
  level = mp$Levels[i]
  if (mp$Topic.Model[i]=="Independent trees"){
    labels=append(labels, bquote("T"^.(level)))
  } else{
    if (mp$Dominant.Mode[i]=="Genes"){
      sub = "G"
    } else{
      sub = "P"
    }
    if (mp$Topic.Set[i]=="Cartesian"){
      sub = paste0(sub,"C")
    } else{
      sub = paste0(sub,"L")
    }
    labels=append(labels, bquote("P"[.(sub)]^.(level)))
  }
}
mp$labels=unlist(labels)
mp$col=as.factor(strsplit(toString(labels),", ")[[1]])

png(filename="~/Documents/Northwestern/Research/Code/plots/cancerLL.png",
    width=4, height=4, units='in', res=480)
par(cex=0.75)
plot(mp$Total.Topics, mp$Mean, type="n", log="x", xlab="Total Topics (log scale)",
     ylab="LL", main="Cancer Log-Likelihood")
text(mp$Total.Topics, mp$Mean, as.expression(mp$labels), cex=.75)
dev.off()

library(ggplot2)
library(ggrepel)
ggp = ggplot(mp, aes(Total.Topics, Mean, color=col))
ggp = ggp + geom_line() + scale_x_log10() 
ggp = ggp + theme_bw() + theme(legend.position = "none")
ggp = ggp + ggtitle("Cancer Model Log-Likelihood")
ggp = ggp + xlab("Total Topics (log scale)") + ylab("Log-Likelihood")
ggp = ggp + theme(plot.title=element_text(hjust = 0.5))
ggp = ggp + geom_text_repel(aes(label=labels), point.padding=0.01,
                             segment.color='grey50', parse=T, data=subset(mp, !duplicated(mp$labels)))
#ggsave(filename="~/Documents/Northwestern/Research/Code/plots/cancerLLg.png", ggp)
ggsave(filename="~/Documents/Northwestern/Research/Code/plots/cancerLLgl.png", ggp)