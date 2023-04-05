### The code implements IPW, outcome regression and DR estimators for the simulation study when both models are correctly specified.   

rm(list=ls())
library(MASS)
library(GoFKernel)
library(tidyverse)

#### read data
dataset_iteration <- 0
folder <- paste0('./experiments/cont_features/dataset_', (2020 + 1000 * dataset_iteration))
X <- read.csv(paste0(folder, '/X.csv'))
A <- X$A
Y <- read.csv(paste0(folder, '/Y.csv'))
X[paste0('X_', seq(0, ncol(X) - 2))] <- X[paste0('X_', seq(0, ncol(X) - 2))]^2
print(head(X))
# Fit propensity score model : misspecify model
print('fitting propensity score...')
logisticfit<-glm('A ~ .', data = X, family=binomial(link='logit'))
pihat<-logisticfit$fitted.values

n <- nrow(X)
lgrid <- ncol(Y)
grid <- seq(0, 1, length.out = lgrid)
identity <- grid
Yinverse<-matrix(NA, n, lgrid)

estimatemu0LY<-estimatemu0IPWcomponent1<-estimatemu0IPWcomponent2<-matrix(NA, n, lgrid)
for (i in 1:n){
    estimatemu0Yinversefunction<-function(u){
        tempindex<-floor(u*(lgrid-1))+1
        if (u>0 & u<1){
        estimatemu0Yinversefunction<-Yinverse[i,tempindex]+(u-grid[tempindex])*(Yinverse[i, tempindex+1]-Yinverse[i, tempindex])*(lgrid-1)
        } else if (u<=0){
        estimatemu0Yinversefunction<-0
        } else {
        estimatemu0Yinversefunction<-1
        }	
    }
    estimatemu0LY[i,]<-sapply(identity, estimatemu0Yinversefunction)
    estimatemu0IPWcomponent1[i,]<-A[i]*estimatemu0LY[i,]/pihat[i]
    estimatemu0IPWcomponent2[i,]<-(1-A[i])*estimatemu0LY[i,]/(1-pihat[i])
}

### IPW estimate
estimatemu0IPWpsi1<-apply(estimatemu0IPWcomponent1, 2, mean)
estimatemu0IPWpsi0<-apply(estimatemu0IPWcomponent2, 2, mean)

IPWmu0inverse<-estimatemu0IPWpsi0


#plot(IPWtxeffect, type="l")

### regression estimate
mu0lmfit_df <- X
mu0lmfit_df$estimatemu0LY <- estimatemu0LY
print(estimatemu0LY)
print('fitting regression estimate...')
estimatemu0lmfit<-lm('estimatemu0LY ~ .', data = mu0lmfit_df)

newdesign1<-cbind(rep(1,n), rep(1, n), X)
newdesign0<-cbind(rep(1,n), rep(0, n), X)

estimatemu0m1fit<-newdesign1%*%estimatemu0lmfit$coefficients
estimatemu0m0fit<-newdesign0%*%estimatemu0lmfit$coefficients

estimatemu0outcomepsi1<-apply(estimatemu0m1fit, 2, mean)
estimatemu0outcomepsi0<-apply(estimatemu0m0fit, 2, mean)

outcomemu0inverse<-estimatemu0outcomepsi0

estimatemu0DRcomponent1<-estimatemu0DRcomponent2<-matrix(NA, n, lgrid)
for (i in 1:n){
    estimatemu0DRcomponent1[i, ]<-estimatemu0IPWcomponent1[i,]-(A[i]/pihat[i]-1)*estimatemu0m1fit[i,]
    estimatemu0DRcomponent2[i, ]<-estimatemu0IPWcomponent2[i,]-((1-A[i])/(1-pihat[i])-1)*estimatemu0m0fit[i,]
}

estimatemu0DRpsi1<-apply(estimatemu0DRcomponent1, 2, mean)
estimatemu0DRpsi0<-apply(estimatemu0DRcomponent2, 2, mean)

DRmu0inverse<-estimatemu0DRpsi0

### linear interpolation 
IPWmu0inversefunction<-function(u){
tempindex<-floor(u*(lgrid-1))+1
if (u>0 & u<1){
    IPWmu0inversefunction<-IPWmu0inverse[tempindex]+(u-grid[tempindex])*(IPWmu0inverse[tempindex+1]-IPWmu0inverse[tempindex])*(lgrid-1)
} else if (u<=0) {
    IPWmu0inversefunction<-0
} else {
    IPWmu0inversefunction<-1
}	
}

### G bar function, mu0 function
IPWmu0inversefun<-inverse(IPWmu0inversefunction, lower=0, upper=1)

### G bar，Estimated Frechet mean (all checked correct, see testhatmu.r)
IPWhatmu0<-sapply(grid, IPWmu0inversefun)

### linear interpolation 
outcomemu0inversefunction<-function(u){
tempindex<-floor(u*(lgrid-1))+1
if (u>0 & u<1){
    outcomemu0inversefunction<-outcomemu0inverse[tempindex]+(u-grid[tempindex])*(outcomemu0inverse[tempindex+1]-outcomemu0inverse[tempindex])*(lgrid-1)
} else if (u<=0) {
    outcomemu0inversefunction<-0
} else {
    outcomemu0inversefunction<-1
}	
}

### G bar function, mu0 function
outcomemu0inversefun<-inverse(outcomemu0inversefunction, lower=0, upper=1)

### G bar，Estimated Frechet mean (all checked correct, see testhatmu.r)
outcomehatmu0<-sapply(grid, outcomemu0inversefun)


### linear interpolation 
DRmu0inversefunction<-function(u){
tempindex<-floor(u*(lgrid-1))+1
if (u>0 & u<1){
    DRmu0inversefunction<-DRmu0inverse[tempindex]+(u-grid[tempindex])*(DRmu0inverse[tempindex+1]-DRmu0inverse[tempindex])*(lgrid-1)
} else if (u<=0) {
    DRmu0inversefunction<-0
} else {
    DRmu0inversefunction<-1
}	
}

### G bar function, mu0 function
DRmu0inversefun<-inverse(DRmu0inversefunction, lower=0, upper=1)

### G bar，Estimated Frechet mean (all checked correct, see testhatmu.r)
DRhatmu0<-sapply(grid, DRmu0inversefun)


# Fit propensity score model
logisticfit<-glm('A~.', data = X, family=binomial(link='logit'))
pihat<-logisticfit$fitted.values

IPWLY<-outcomeLY<-DRLY<-IPWcomponent1<-IPWcomponent2<-DRIPWcomponent1<-DRIPWcomponent2<-matrix(NA, n, lgrid)
for (i in 1:n){
Yinversefunction<-function(u){
    tempindex<-floor(u*(lgrid-1))+1
    if (u!=1){
    Yinversefunction<-Yinverse[i,tempindex]+(u-grid[tempindex])*(Yinverse[i, tempindex+1]-Yinverse[i, tempindex])*(lgrid-1)
    } else {
    Yinversefunction<-1
    }	
}

IPWLY[i,]<-sapply(IPWhatmu0, Yinversefunction)
outcomeLY[i,]<-sapply(outcomehatmu0, Yinversefunction)
DRLY[i,]<-sapply(DRhatmu0, Yinversefunction)

IPWcomponent1[i,]<-A[i]*IPWLY[i,]/pihat[i]
IPWcomponent2[i,]<-(1-A[i])*IPWLY[i,]/(1-pihat[i])

DRIPWcomponent1[i,]<-A[i]*DRLY[i,]/pihat[i]
DRIPWcomponent2[i,]<-(1-A[i])*DRLY[i,]/(1-pihat[i])
}

# ### IPW estimate
# IPWtxeffect<-apply(IPWcomponent1, 2, mean)-apply(IPWcomponent2, 2, mean)


# ### IPW estimate
# IPWtxeffect<-apply(IPWcomponent1, 2, mean)-apply(IPWcomponent2, 2, mean)

# ### regression estimate
# outcomelmfit<-lm(outcomeLY~A+X)
# outcometxeffect<-outcomelmfit$coefficients[2,]

# DRlmfit<-lm(DRLY~A+X)

# newdesign1<-cbind(rep(1,n), rep(1, n), X)
# newdesign0<-cbind(rep(1,n), rep(0, n), X)

# m1fit<-newdesign1%*%DRlmfit$coefficients
# m0fit<-newdesign0%*%DRlmfit$coefficients

# DRcomponent1<-DRcomponent2<-matrix(NA, n, lgrid)
# for (i in 1:n){
# DRcomponent1[i, ]<-DRIPWcomponent1[i,]-(A[i]/pihat[i]-1)*m1fit[i,]
# DRcomponent2[i, ]<-DRIPWcomponent2[i,]-((1-A[i])/(1-pihat[i])-1)*m0fit[i,]
# }

# DRtxeffect<-apply(DRcomponent1, 2, mean)-apply(DRcomponent2, 2, mean)


# DRpsi1<-apply(DRcomponent1, 2, mean)
# DRpsi0<-apply(DRcomponent2, 2, mean)

# DRindividual<-DRcomponent1-DRcomponent2

# calsequence<-seq(5, lgrid-1, 10)

# calDRindividual<-DRindividual[,calsequence]

# covcalDRindividual<-cov(calDRindividual)
# meancalDRindividual<-apply(calDRindividual, 2, mean)

# generatenormalvariable<-mvrnorm(1000, rep(0, length(calsequence)), covcalDRindividual)

# maxgeneratenormalvariable<-apply(abs(generatenormalvariable), 1, max)
# sortmaxgeneratenormalvariable<-sort(maxgeneratenormalvariable)
# DRband<-sortmaxgeneratenormalvariable[950]/sqrt(n)

# DRhigherband<-DRtxeffect+DRband
# DRlowerband<-DRtxeffect-DRband


# highercoverageindicator<-sum(DRhigherband<truetreatmentvalue)
# lowercoverageindicator<-sum(DRlowerband>truetreatmentvalue)

# if (sum(highercoverageindicator+lowercoverageindicator)==0){
# DRcoverage[s]<-1
# } else {
# DRcoverage[s]<-0
# }



# IPWtransportvalue<-sapply(truemu0, IPWmu0inversefunction)


# IPWtxeffectfunction<-function(u){
# tempindex<-floor(u*(lgrid-1))+1
# if (u>0 & u<1){
#     IPWtxeffectfunction<-IPWtxeffect[tempindex]+(u-grid[tempindex])*(IPWtxeffect[tempindex+1]-IPWtxeffect[tempindex])*(lgrid-1)
# } else {
#     IPWtxeffectfunction<-0
# }
# }

# IPWparralleleffect<-sapply(IPWtransportvalue, IPWtxeffectfunction)


# outcometransportvalue<-sapply(truemu0, outcomemu0inversefunction)


# outcometxeffectfunction<-function(u){
# tempindex<-floor(u*(lgrid-1))+1
# if (u>0 & u<1){
#     outcometxeffectfunction<-outcometxeffect[tempindex]+(u-grid[tempindex])*(outcometxeffect[tempindex+1]-outcometxeffect[tempindex])*(lgrid-1)
# } else {
#     outcometxeffectfunction<-0
# }
# }

# outcomeparralleleffect<-sapply(outcometransportvalue, outcometxeffectfunction)


# DRtransportvalue<-sapply(truemu0, DRmu0inversefunction)

# DRtxeffectfunction<-function(u){
# tempindex<-floor(u*(lgrid-1))+1
# if (u>0 & u<1){
#     DRtxeffectfunction<-DRtxeffect[tempindex]+(u-grid[tempindex])*(DRtxeffect[tempindex+1]-DRtxeffect[tempindex])*(lgrid-1)
# } else {
#     DRtxeffectfunction<-0
# }
# }

# DRparralleleffect<-sapply(DRtransportvalue, DRtxeffectfunction)

# IPWbias[s,]<-IPWparralleleffect-truetreatmentvalue

# outcomebias[s,]<-outcomeparralleleffect-truetreatmentvalue

# DRbias[s, ]<-DRparralleleffect-truetreatmentvalue

# IPWMSE[s]<-sqrt(mean((IPWbias[s,])^2))
# outcomeMSE[s]<-sqrt(mean((outcomebias[s,])^2))
# DRMSE[s]<-sqrt(mean((DRbias[s,])^2))

# tempsequence<-seq(101, 901, by=100)
# IPWmedianbias[s,]<-IPWparralleleffect[tempsequence]-truetreatmentvalue[tempsequence]

# outcomemedianbias[s,]<-outcomeparralleleffect[tempsequence]-truetreatmentvalue[tempsequence]

# DRmedianbias[s,]<-DRparralleleffect[tempsequence]-truetreatmentvalue[tempsequence]

# #plot(truetreatmentvalue)
# #lines(outcomeparralleleffect)
# #lines(DRparralleleffect)
# #lines(IPWparralleleffect)

