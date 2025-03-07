dir = "/Volumes/fsmresfiles/PrevMed/HBMI/LYG/LabWorkstationMount/asd/vaRData_uw/"; #folder
files = list.files(path=dir, pattern = "\\.RData$"); #get RData files

id=c();
refGen=c();

for(i in 1:length(files)){
  load(paste0(dir,files[i])); #load files
  id=c(id,as.character(va$"VCF.ID")); #get id
  refGen=c(refGen,as.character(va$"Gene.refGen")); #get gene names
  
  #subset to get only unique ids
  u=which(!duplicated(id));
  id=id[u];
  refGen=refGen[u];
}

rm(va);

geneRef=matrix(NA,nrow=length(id),ncol=2);
colnames(geneRef)=c("ID","refGen");
geneRef[,1]=id;
geneRef[,2]=refGen;

semicolon=grep(';', geneRef$"refGen", value=TRUE);
if(length(semicolon)!=0){
  clean=which(geneRef$"refGen" %in% semicolon);
  for(i in 1:length(semicolon)){
    new=strsplit(semicolon[i],";")[[1]];
    geneRef[clean[i],2]=new[1];
    if(new[1]!=new[2]){
      geneRef=rbind(geneRef,c(geneRef[clean[i],1],new[2]));
    }
  }
}
comma=grep(',', geneRef$"refGen", value=TRUE);
if(length(comma)!=0){
  clean=which(geneRef$"refGen" %in% comma);
  for(i in 1:length(comma)){
    new=strsplit(comma[i],",")[[1]];
    geneRef[clean[i],2]=new[1];
    if(new[1]!=new[2]){
      geneRef=rbind(geneRef,c(geneRef[clean[i],1],new[2]));
    }
  }
}

write.csv(geneRef, "/Users/adamsandler/Documents/Northwestern/Research/Code/geneRef.csv"); #export to csv
#geneRef <- read.csv("~/Documents/Northwestern/Research/Code/geneRef.csv", row.names = 1, header= T, check.names=F, stringsAsFactors = F); #import from csv
