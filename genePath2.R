#t=read.table("https://reactome.org/download/current/Ensembl2Reactome.txt", sep="\t", stringsAsFactor = F);
library(readtext); #load package
t=readtext("https://reactome.org/download/current/Ensembl2Reactome.txt")[1,2]; #read text file
t=strsplit(t,"\n")[[1]]; #split on returns
t=strsplit(t,"\t"); #split on tabs
t=matrix(unlist(t), nrow=length(t), byrow=T); #unlist as matrix
t=subset(t, t[,6]=="Homo sapiens"); #subset to get humans

write.csv(t, "/Users/adamsandler/Documents/Northwestern/Research/Code/genePath2.csv"); #export to csv
#genePath <- read.csv("~/Documents/Northwestern/Research/Code/genePath2.csv", header= T, check.names=F); #import from csv
