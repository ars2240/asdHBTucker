geneRef <- read.csv("~/Documents/Northwestern/Research/Code/geneRef2.csv", row.names = 1, header= T, check.names=F, stringsAsFactors = F); #import from csv
genePath <- read.csv("~/Documents/Northwestern/Research/Code/genePathND4.csv", row.names = 1, header= T, check.names=F, stringsAsFactors = F); #import from csv
asd <- read.csv("~/Documents/Northwestern/Research/Code/asd.csv", row.names = 1, header= T, check.names=F); #import from csv

#subset to get only variants, genes, and pathways in all 3 tables
geneRef=subset(geneRef, geneRef$ID %in% colnames(asd));
#geneRef=geneRef[which(!duplicated(geneRef$ID)),]
#write.csv(geneRef, "/Users/adamsandler/Documents/Northwestern/Research/Code/geneRef2.csv"); #export to csv
geneRef=subset(geneRef, geneRef$refGen %in% genePath[,1]);
genePath=genePath[genePath[,1] %in% geneRef$refGen,];
asd=asd[,colnames(asd) %in% geneRef$ID];

#create list of unique pathways
pathways=unique(genePath[,2]);
#genePath=subset(genePath,!duplicated(genePath[,2]))
genePath=genePath[order(genePath[,2]),]
pathways=pathways[order(pathways)]; #sort
#write.csv(genePath, "/Users/adamsandler/Documents/Northwestern/Research/Code/genePath3.csv")

#library(rTensor); #load package
#library(tensorr);

indices=c();
indices[1]=nrow(asd); #number of patients
indices[2]=ncol(asd); #number of genetic variants
indices[3]=length(pathways); #number of pathways
#asdTens=array(rep(0,prod(indices)), dim=indices);
#asdTens=as.tensor(asdTens);

#initialize sparse representation
sparse=which(asd!=0,arr.ind = T); #nonzero elements
rownames(sparse)=1:nrow(sparse);
sparse=data.frame(sparse);
sparse$pathway=NA;
sparse$value=asd[which(asd!=0,arr.ind = T)]; #value
for(i in 1:indices[2]){
  spRows=which(sparse$col==i); #rows in sparse matrix
  genes=which(geneRef$ID==colnames(asd)[i]); #get gene names
  paths=which(genePath[,1] %in% geneRef$refGen[genes]); #get pathways
  k=which(pathways %in% genePath[paths,2]); #get number of pathway
  sparse$pathway[spRows]=k[1];
  if(length(k)>1){
    for(j in 2:length(k)){
      temp=sparse[spRows,];
      temp$pathway=k[j];
      sparse=rbind(sparse,temp);
    }
  }
}
write.csv(sparse, "/Users/adamsandler/Documents/Northwestern/Research/Code/asdSparseND.csv") #export to csv
#sparse <- read.csv("~/Documents/Northwestern/Research/Code/asdSparse.csv", header= T, check.names=F, stringsAsFactors = F); #import from csv

#asdTens=sptensor(as.matrix(t(sparse[1:1000,1:3])),as.numeric(sparse[1:1000,4]),indices)

#source('~/Documents/Northwestern/Research/Code/asdHBTucker.R'); #load functions

