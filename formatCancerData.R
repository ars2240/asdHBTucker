genename = read.csv("/Volumes/fsmresfiles/PrevMed/HBMI/LYG/LabWorkstationMount/tcga_cancer_matrix/genename_with_ENSID.csv", header = T, stringsAsFactors = F) # header?
label = read.csv("/Volumes/fsmresfiles/PrevMed/HBMI/LYG/LabWorkstationMount/tcga_cancer_matrix/label.csv", row.names = 1, header= T, check.names=F, stringsAsFactors = F)
number = read.csv("/Volumes/fsmresfiles/PrevMed/HBMI/LYG/LabWorkstationMount/tcga_cancer_matrix/number.csv", row.names = 1, header= T, check.names=F, stringsAsFactors = F)

cList=c("BRCA", "LUAD", "LUSC", "PRAD", "COAD", "READ") #list of cancers studied
label=subset(label, cancer %in% cList)

genename=subset(genename, !is.na(ensembl_gene_id))

#subset tables with patients in both
label$SampleLabel=substr(label$SampleName,1,4)
label=label[label$SampleLabel %in% rownames(number),]
number=number[rownames(number) %in% label$SampleLabel,]

#subset genes
number=number[,colnames(number) %in% genename$V1]
number=number[,colSums(number)>0]
genename=genename[genename$V1 %in% colnames(number),]

#re-name same cancers
label$cancer[label$cancer=="LUSC"]="LUAD"
label$cancer[label$cancer=="READ"]="COAD"

#re-order
number=number[order(rownames(number)),]
label=label[order(label$SampleLabel),]
rownames(label)=label$SampleLabel

genepath = read.csv("/Users/adamsandler/Documents/Northwestern/Research/Code/genePath2.csv", row.names = 1, header= T, check.names=F, stringsAsFactors = F)

#subset genes with pathway information
genepath=genepath[genepath$V1 %in% genename$ensembl_gene_id,]
genename=genename[genename$ensembl_gene_id %in% genepath$V1,]
number=number[,colnames(number) %in% genename$V1]
number=number[rowSums(number)>0,]
print(sum(colSums(number)==0))
# need to subset by colSums>0 (check) and may need to shrink genename & gene path

#subset tables with patients in both
label=label[label$SampleLabel %in% rownames(number),]

#cancer classes
cancerLabel=label$cancer
cancers=unique(cancerLabel)
cancers=cancers[order(cancers)]
cancerLabel=match(cancerLabel, cancers)-1

write.csv(cancerLabel, "/Users/adamsandler/Documents/Northwestern/Research/Code/cancerLabel.csv")

#re-order genes
number=number[,order(colnames(number))]
genename=genename[order(genename$V1),]

write.csv(number, "/Users/adamsandler/Documents/Northwestern/Research/Code/cancerNumber.csv")
write.csv(genename, "/Users/adamsandler/Documents/Northwestern/Research/Code/cancerGeneName.csv")

#sparse representation
sparse=which(number!=0,arr.ind = T)
rownames(sparse)=1:nrow(sparse)
colnames(sparse)=c("patient","gene")
sparse=data.frame(sparse)
sparse$value=number[which(number!=0,arr.ind = T)] #value
write.csv(sparse, "/Users/adamsandler/Documents/Northwestern/Research/Code/cancerSparseGenes.csv")

#get unique pathways
pathways=unique(genepath$V2)
pathways=pathways[order(pathways)]

sparse$pathway=NA
sparse=sparse[,c("patient","gene","pathway","value")]
for(i in 1:nrow(genename)){
  spRows=which(sparse$gene==i) #rows in sparse matrix
  gene=genename$V2[i] #get gene ID
  paths=which(genepath[,1]==gene); #get pathways
  k=which(pathways %in% genepath[paths,2]); #get number of pathway
  sparse$pathway[spRows]=k[1];
  if(length(k)>1){
    for(j in 2:length(k)){
      temp=sparse[spRows,];
      temp$pathway=k[j];
      sparse=rbind(sparse,temp);
    }
  }
}

write.csv(sparse, "/Users/adamsandler/Documents/Northwestern/Research/Code/cancerSparse.csv") #export to csv