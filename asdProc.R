dir = "/Volumes/fsmresfiles/PrevMed/HBMI/LYG/LabWorkstationMount/asd/mGT_uw/"; #folder
files = list.files(path=dir, pattern = "\\.RData$"); #get RData files

genes = c(); #vector of genes
patients = c(); #vector of patients
mgtData = list(NA); #list of mGT data files

for(i in 1:length(files)){
  load(paste0(dir,files[i])); #load files
  mgtData[[i]] = mGT; #store mGT data file
  genes = unique(c(genes,colnames(mGT))); #add unique genes
  patients = c(patients,rownames(mGT)); #add patients
}

genes=genes[order(genes)]; #sort genes
asd = matrix(0, length(patients), length(genes)); #create matrix
colnames(asd)=genes; #set column names to genes
rownames(asd)=patients; #set row names to patients

# sink(paste("~/Documents/Northwestern/Research/Code/geneDifPos.txt", sep=""))
# t=0
for(i in 1:length(files)){
  split=strsplit(mgtData[[i]],"/"); #split elemenets along /
  nPat=nrow(mgtData[[i]]); #number of patients
  g=colnames(mgtData[[i]]); #genes
  p=rownames(mgtData[[i]]); #patients
  l=length(split); #total elements
  row=(1:l-1) %% nPat + 1; #row number
  col=ceiling(1:l/nPat); #column number
  for (j in 1:l){
    if(sum(is.na(split[[j]]))>0 || length(split[[j]])!=2){
      count=sum(as.numeric(split[[j]])>0,na.rm=T); #count of mutations
    } else{
      if(split[[j]][1]>0 && split[[j]][2]>0){
        count=(split[[j]][1]==split[[j]][2])+1
      } else {
        count=sum(as.numeric(split[[j]])>0,na.rm=T); #count of mutations
      }
    }
    # if(count==2 && as.numeric(split[[j]][1])!=as.numeric(split[[j]][2])){
    #   print(split[[j]])
    #   print(p[row[j]])
    #   print(g[col[j]])
    #   cat('\n')
    #   t=t+1
    # }
    asd[p[row[j]],g[col[j]]]=count; #add count to table
  }
}
# sink()
# print(t)

rm(mgtData);
rm(mGT);
write.csv(asd, "/Users/adamsandler/Documents/Northwestern/Research/Code/asd.csv") #export to csv
#asd <- read.csv("~/Documents/Northwestern/Research/Code/asd.csv", row.names = 1, header= T, check.names=F); #import from csv
