#read in text file
t=read.table("/Users/adamsandler/Documents/Northwestern/Research/Code/network.txt", sep="\n");
t=as.character(t[,1]); #convert to vector

#cleanup
clean=which(nchar(t)>100);
for(i in 1:length(clean)){
  lines=strsplit(t[clean[i]],"\n")[[1]];
  if(length(lines)>1){
    t=c(t[1:(clean[i]-1)],lines,t[(clean[i]+1):length(t)]);
    clean=clean+length(lines)-1;
  }
}

pStart=which(startsWith(t, "ENTRY")); #rows with pathways
gStart=which(startsWith(t, "GENE")); #rows were genes start

pathways=c();
genesInPathways=list(NA);
genes=c();

for(i in 1:length(pStart)){
  ps=pStart[i];
  
  #next pathway
  if(i<length(pStart)){
    nextP=pStart[i+1];
  }
  else{
    nextP=length(t);
  }
  
  #get label of pathway
  p=strsplit(t[ps], " ")[[1]];
  p=p[p!=""];
  p=p[2];
  
  pathways=c(pathways,p); #add to list
  
  g=gStart[which(gStart>ps & gStart<nextP)]; #gene position
  gs=g;
  
  #boolean for loop
  if(length(g)!=0){
    bool=T;
  }
  else{
    bool=F;
  }
  
  while(bool==T){
    #get gene name
    gene=strsplit(t[g]," ")[[1]];
    gene=gene[gene!=""];
    gene=gene[endsWith(gene,";")];
    gene=substr(gene,1,nchar(gene)-1); #remove ;
    
    #add to vectors
    genes=c(genes,gene);
    if(g==gs){
      genesInPathways[[i]]=c(gene);
    }
    else{
      genesInPathways[[i]]=c(genesInPathways[[i]],gene);
    }
    
    g=g+1; #increment
    bool=startsWith(t[g]," ") #check if still on gene
  }
  
  genes=unique(genes);
}

genes=genes[order(genes)];

#create matrix of genes and pathways
genePath=matrix(0,nrow=length(pathways),ncol=length(genes));
rownames(genePath)=pathways;
colnames(genePath)=genes;

for(i in 1:length(pathways)){
  genePath[i,which(genes %in% genesInPathways[[i]])]=1;
}


write.csv(genePath, "/Users/adamsandler/Documents/Northwestern/Research/Code/genePath.csv"); #export to csv
#genePath <- read.csv("~/Documents/Northwestern/Research/Code/genePath.csv", row.names = 1, header= T, check.names=F); #import from csv
