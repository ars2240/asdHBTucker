geneRef <- read.csv("~/Documents/Northwestern/Research/Code/geneRef.csv", row.names = 1, header= T, check.names=F, stringsAsFactors = F); #import from csv
genePath <- read.csv("~/Documents/Northwestern/Research/Code/genePath.csv", row.names = 1, header= T, check.names=F); #import from csv
asd <- read.csv("~/Documents/Northwestern/Research/Code/asd.csv", row.names = 1, header= T, check.names=F); #import from csv

#subset to get only variants, genes, and pathways in all 3 tables
geneRef=subset(geneRef, geneRef$ID %in% colnames(asd));
#geneRef=geneRef[which(!duplicated(geneRef$ID)),]
#write.csv(geneRef, "/Users/adamsandler/Documents/Northwestern/Research/Code/geneRef2.csv"); #export to csv
geneRef=subset(geneRef, geneRef$refGen %in% colnames(genePath));
genePath=genePath[,colnames(genePath) %in% geneRef$refGen];
asd=asd[,colnames(asd) %in% geneRef$ID];
genePath=genePath[rowSums(genePath)>0,];
