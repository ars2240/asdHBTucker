library(mRMRe) # load package
nfeat=100 # select number of features

# load data set
asdData <- read.csv("~/Documents/Northwestern/Research/Code/asdHBTucker_gam0.1.csv", header=F, stringsAsFactors=F)

# format as numerical
for(i in 1:ncol(asdData)){
  asdData[,i]=as.numeric(asdData[,i])
}

dd <- mRMR.data(data = asdData)
filter <- mRMR.classic(data = dd, target_indices = c(1), feature_count = nfeat)  #run mRMR

# get columns and subset
cols <- c(solutions(filter)$'1')
asdData <- asdData[,cols]

# write csvs
write.csv(asdData, paste("~/Documents/Northwestern/Research/Code/asdHBTucker_",nfeat,"_mRMR.csv",sep=""))
write.csv(cols, paste("~/Documents/Northwestern/Research/Code/asdHBTucker_",nfeat,"_mRMR_cols.csv",sep=""))