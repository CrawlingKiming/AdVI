######################################################################################
##	Simulation of Common-Cold example
######################################################################################
set.seed(1359)
rm(list=ls())
library(latex2exp)
library(deSolve)
library(coda)
SIR <- function(t, x, parameters){
  ##Inputs:
  #t : time
  #x : state variables
  #parameters : parameter vector
  S <- x[1]
  I <- x[2]
  R <- x[3]
  N <- S + I + R
  #The with statement means we can use the parameter names as in parameters
  with( as.list(parameters), {
    dS <- -beta * S * I / N
    dI <- beta * S * I / N - gamma * I
    dR <- gamma * I
    res <- c(dS, dI, dR)
    list(res)
  }
  )
}


run_model <- function(S0, beta, gamma){
  #Function to run model with dede or lsoda
  out <- lsoda(func = SIR,
               y = c(S = S0, I = 1, R = 0),
               parms = c(beta = beta, gamma = gamma),
               times = seq(1, 21, by = 1))
  return(data.frame(out[, c("I", "R")]))
}

### Test ##################################################
D_ = run_model(38, 0.5, 0.5)
par(mfrow=c(1,1),cex=2,mar=c(5,4,4,1)+0.1)

plot(D_[,2])
lines(D_[,1])
  
### Set ####################################################
obs_I = c(1,1,3,7,6,10,13,13,14,14,17,10,6,6,4,3,1,1,1,1,0)
obs_R = c(0,0, 0, 0, 5, 7, 8, 13, 13, 16, 16, 24, 30, 31,33, 34, 36, 36, 36,36, 37)
#obs_R = obs_RR[2:22] - obs_RR[1:21]
obs = cbind((obs_I), (obs_R))
t_length = 21 
# Construct MCMC 

## Construct fts for running 
S0 = 40
beta = 0.4
gamma = 0.4

loglik = function(pars) {
  with(as.list(pars), {
    if( 	S0 < 37 || S0 > 100 ||
         beta < 0.0 || beta > 3.0 ||
         gamma < 0.0 || gamma > 3.0) { return(-1e10) } # boundaries 
    t_length = 21
    t = t_length 
    #print(t)
    
    out = run_model(S0, beta, gamma)
    
    out_severed_I = out[1:t, 1]
    out_severed_R = (out[1:t,2])
    obs_severed_I = obs[1:t, 1]
    obs_severed_R = obs[1:t, 2]#,dgamma(beta, shape = 1, rate=1, log=TRUE)
    
    sum(dpois(obs_severed_I, lambda= out_severed_I, log=TRUE), dpois(obs_severed_R, lambda= out_severed_R, log=TRUE))#, dgamma(beta, shape = 1.5, scale=1, log=TRUE), dgamma(gamma, shape=1, scale=1))#, dpois(obs_severed_R[t], lambda = out_severed_R[t], log=TRUE))#, dgamma(sig, 1,1, log=TRUE),dnorm(obs_severed_R, mean =out_severed_R, sd=sig, log=TRUE))#,, dpois(obs_severed_R,lambda = out_severed_R, log=TRUE)
  })
}
#dnorm(out_severed_I,mean=obs_severed_I, sd=0.5, log=TRUE) + dnorm(out_severed_R,mean=obs_severed_R, sd=0.5, log=TRUE)
t_length = 2
checkpar = c(S0=38, beta = 0.5, gamma = 0.5)
loglik(pars=checkpar)

######################################################################################

# run MCMC 

t_length = 21


library("readxl")
library(adaptMCMC)

init.pars = c(S0=40, beta = 0.2, gamma = 0.3)
nn = 50000 #Number of Samples 
start.time <- Sys.time()
out.mcmc <- MCMC(p=loglik, n=nn, init=init.pars, scale=init.pars/2, adapt=TRUE, acc.rate=.2)
end.time <- Sys.time()
time.taken.fullmcmc <- end.time - start.time
t_length=21

burnt_mcmc = out.mcmc$samples[-(seq(1, 2000)),]
save(burnt_mcmc, file="burnt_mcmc.Rdata")
load("burnt_mcmc.Rdata")


par(mfrow=c(1,3), cex=1.75, mar=c(4,2,1,1)+0.1) 
hist(burnt_mcmc[,1], 30, col="#AEAEAE", xlab=expression("S0"), main="", ylab="")#, #xlim = c(37, 60))
hist(burnt_mcmc[,2], 30, col="#AEAEAE", xlab=expression(beta), main="", ylab="")#, xlim = c(0.2, .4))
hist(burnt_mcmc[,3], 30, col="#AEAEAE", xlab=expression(gamma), main="", ylab="")#, xlim = c(0.3, 0.5))


effectiveSize(burnt_mcmc)

hist(burnt_mcmc[,3])
ts.plot(burnt_mcmc[,1])

ATVI = read.csv(file="BTAT_0.csv", header = FALSE)
VI = read.csv(file="BFAF.csv", header = FALSE)
Coupla = read.csv(file="Coupla.csv", header = FALSE)
MCMC <- burnt_mcmc#[sample(nrow(burnt_mcmc), nrow(ATVI)), ]


### GETING PARAMETERS RESULTS ##
png(filename="param_plots.png", width=1040, height=480)
par(mfrow=c(1,3))
plot(density(MCMC[,1], from=37, to=100), col="black", ylim=c(0.0, 0.45),lwd=2,lty=1, main = TeX("$S_{0}$"), cex.lab = 2.2, cex.main = 2.5, xlab="", ylab="", cex.axis=1.3,  xlim=c(37,50))
lines(density(ATVI[,3], from=37, to=100), col="red",lwd=2,lty=2)
lines(density(VI[,3], from=37, to=100), col="blue",lwd=2,lty=3)
lines(density(Coupla[,3], from=37, to=100), col="green",lwd=2,lty=4)
legend("topright", legend = c("MCMC","ATVI", "VI (NF)", "VI (Copula)"), col = c("black", "red", "blue", "green"), lty = c(1,2,3,4), lwd = 2, cex = 1.2)
plot(density(MCMC[,2], from=0., to=3.0), col="black", ylim=c(0.0, 22.0),lwd=2,lty=1, main = TeX("$beta$"), cex.lab = 2.2, cex.main = 2.5, xlab="", ylab="", cex.axis=1.3, xlim=c(0.7,1.1))
lines(density(ATVI[,1], from=0, to=3.0), col="red",lwd=2,lty=2)
lines(density(VI[,1], from=0, to=3.0), col="blue",lwd=2,lty=3)
lines(density(Coupla[,1], from=0, to=3.0), col="green",lwd=2,lty=4)
legend("topright", legend = c("MCMC","ATVI", "VI (NF)", "VI (Copula)"), col = c("black", "red", "blue", "green"), lty = c(1,2,3,4), lwd = 2, cex = 1.2)
plot(density(MCMC[,3], from=0., to=3.0), col="black", ylim=c(0.0, 25.0),lwd=2,lty=1, main = TeX("$gamma$"), cex.lab = 2.2, cex.main = 2.5, xlab="", ylab="", cex.axis=1.3, xlim=c(0.17,0.4))
lines(density(ATVI[,2], from=0, to=3.0), col="red",lwd=2,lty=2)
lines(density(VI[,2], from=0, to=3.0), col="blue",lwd=2,lty=3)
lines(density(Coupla[,2], from=0, to=3.0), col="green",lwd=2,lty=4)
legend("topright", legend = c("MCMC","ATVI", "VI (NF)", "VI (Copula)"), col = c("black", "red", "blue", "green"), lty = c(1,2,3,4), lwd = 2, cex = 1.2)
dev.off()
HPDinterval(as.mcmc(MCMC))


data = data.frame(ATVI,VI,Coupla)

HPDinterval(as.mcmc(data))
apply(data, 2, median)


### GETTING FORWARD RESULTS ##
MCMC = burnt_mcmc
param_mat = ATVI
get_runs = function(param_mat, n = 1000){
  set.seed(1)
  sampled_rows = sample(x = 1:nrow(param_mat), size = n, replace = FALSE)
  param_mat = param_mat[sampled_rows, ]
  colnames(param_mat) = c()
  j = 1
  res = run_model(param_mat[j,3],param_mat[j,1],param_mat[j,2])
  I = res[,1]
  R = res[,2]
  for(j in 2:n){
    temp = run_model(param_mat[j,3],param_mat[j,1],param_mat[j,2])
    ttemp = matrix(nrow=nrow(temp), ncol=2)
    for(jj in 1:nrow(temp)){
      ttemp[jj,1] = rpois(1, lambda = temp[jj,1])#, sd=1)#, sd=param_mat[j,4])
      ttemp[jj,2] = rpois(1, lambda = temp[jj,2])#, sd=5)
    }
    I = cbind(I, ttemp[,1])
    R = cbind(R, ttemp[,2])
  }
  Ip = coda::HPDinterval(as.mcmc(t(I)))
  Rp = coda::HPDinterval(as.mcmc(t(R)))
  #data.frame()
  mspeI = mean((t(I) - obs[,1])**2)
  mspeR = mean((t(R) - obs[,2])**2)
  tt = data.frame(cbind(Ip, apply(t(I), 2, median)), cbind(Rp,  apply(t(R), 2, median)))
  return(list(tt, mspeI, mspeR))
}
ATVIp = get_runs(as.matrix(ATVI), n=2000)
VIp = get_runs(as.matrix(VI), n=2000)
Gp = get_runs(as.matrix(Coupla), n=2000)
get_runs2 = function(param_mat, n = 1000){
  set.seed(1)
  sampled_rows = sample(x = 1:nrow(param_mat), size = 2000, replace = FALSE)
  param_mat = param_mat[sampled_rows, ]
  colnames(param_mat) = c()
  j = 1
  res = run_model(param_mat[j,1],param_mat[j,2],param_mat[j,3])
  I = res[,1]
  R = res[,2]
  for(j in 2:n){
    temp = run_model(param_mat[j,1],param_mat[j,2],param_mat[j,3])
    ttemp = matrix(nrow=nrow(temp), ncol=2)
    for(jj in 1:nrow(temp)){
      ttemp[jj,1] = rpois(1, lambda = temp[jj,1])#, sd=1)#, sd=param_mat[j,4])
      ttemp[jj,2] = rpois(1, lambda = temp[jj,2])#, sd=5)
    }
    I = cbind(I, ttemp[,1])
    R = cbind(R, ttemp[,2])
  }
  Ip = coda::HPDinterval(as.mcmc(t(I)))
  Rp = coda::HPDinterval(as.mcmc(t(R)))
  mspeI = mean((t(I) - obs[,1])**2)
  mspeR = mean((t(R) - obs[,2])**2)
  tt = data.frame(cbind(Ip, apply(t(I), 2, median)), cbind(Rp,  apply(t(R), 2, median)))
  return(list(tt, mspeI, mspeR))
}
MCMCp = get_runs2((MCMC), n=2000)

## ATVI 
png(filename="ATVI_F.png", width=840, height=480)
temp = ATVIp
par(mfrow=c(1,2))
plot(obs[,1], col="black", ylim=c(0.0, 20),lwd=2,lty=1, main = "Infected", cex.lab = 1.5, cex.main = 2.0, xlab="Time (Days)", ylab="Number", cex.axis=1.3)#,  xlim=c(37,50))
lines(temp[1][[1]][,1], col="red",lwd=2,lty=2)
lines(temp[1][[1]][,2], col="red",lwd=2,lty=2)
lines(temp[1][[1]][,3], col="red",lwd=2,lty=1)

plot(obs[,2], col="black", ylim=c(0.0, 54.0),lwd=2,lty=1, main = "Recovered", cex.lab = 1.5, cex.main = 2.0, xlab="Time (Days)", ylab="Number", cex.axis=1.3)#, xlim=c(0.7,1.1))
lines(temp[1][[1]][,4], col="red",lwd=2,lty=2)
lines(temp[1][[1]][,5], col="red",lwd=2,lty=2)
lines(temp[1][[1]][,6], col="red",lwd=2,lty=1)
dev.off()
#MSPE
print(temp[c(2,3)])
#ATL
mean(sum(temp[1][[1]][,5]-temp[1][[1]][,4], temp[1][[1]][,2]-temp[1][[1]][,1])) / 2
#Data Coverage 
m0  = (temp[1][[1]][,1]<=obs[,1]) & (obs[,1]<=temp[1][[1]][,2])
m1 = (temp[1][[1]][,4]<=obs[,2]) & (obs[,2]<=temp[1][[1]][,5])
(sum(m0)+sum(m1))/42

## VI 
png(filename="VI_F.png", width=840, height=480)
temp = VIp
par(mfrow=c(1,2))
plot(obs[,1], col="black", ylim=c(0.0, 20),lwd=2,lty=1, main = "Infected", cex.lab = 1.5, cex.main = 2.0, xlab="Time (Days)", ylab="Number", cex.axis=1.3)#,  xlim=c(37,50))
lines(temp[1][[1]][,1], col="red",lwd=2,lty=2)
lines(temp[1][[1]][,2], col="red",lwd=2,lty=2)
lines(temp[1][[1]][,3], col="red",lwd=2,lty=1)

plot(obs[,2], col="black", ylim=c(0.0, 54.0),lwd=2,lty=1, main = "Recovered", cex.lab = 1.5, cex.main = 2.0, xlab="Time (Days)", ylab="Number", cex.axis=1.3)#, xlim=c(0.7,1.1))
lines(temp[1][[1]][,4], col="red",lwd=2,lty=2)
lines(temp[1][[1]][,5], col="red",lwd=2,lty=2)
lines(temp[1][[1]][,6], col="red",lwd=2,lty=1)
dev.off()
#MSPE
print(temp[c(2,3)])
#ATL
mean(sum(temp[1][[1]][,5]-temp[1][[1]][,4], temp[1][[1]][,2]-temp[1][[1]][,1])) / 2
#Data Coverage 
m0  = (temp[1][[1]][,1]<=obs[,1]) & (obs[,1]<=temp[1][[1]][,2])
m1 = (temp[1][[1]][,4]<=obs[,2]) & (obs[,2]<=temp[1][[1]][,5])
(sum(m0)+sum(m1))/42

## MCMC
png(filename="MCMC_F.png", width=840, height=480)
temp = MCMCp
par(mfrow=c(1,2))
plot(obs[,1], col="black", ylim=c(0.0, 20),lwd=2,lty=1, main = "Infected", cex.lab = 1.5, cex.main = 2.0, xlab="Time (Days)", ylab="Number", cex.axis=1.3)#,  xlim=c(37,50))
lines(temp[1][[1]][,1], col="red",lwd=2,lty=2)
lines(temp[1][[1]][,2], col="red",lwd=2,lty=2)
lines(temp[1][[1]][,3], col="red",lwd=2,lty=1)

plot(obs[,2], col="black", ylim=c(0.0, 54.0),lwd=2,lty=1, main = "Recovered", cex.lab = 1.5, cex.main = 2.0, xlab="Time (Days)", ylab="Number", cex.axis=1.3)#, xlim=c(0.7,1.1))
lines(temp[1][[1]][,4], col="red",lwd=2,lty=2)
lines(temp[1][[1]][,5], col="red",lwd=2,lty=2)
lines(temp[1][[1]][,6], col="red",lwd=2,lty=1)
dev.off()
#MSPE
print((temp[c(2,3)]))
#ATL
mean(sum(temp[1][[1]][,5]-temp[1][[1]][,4], temp[1][[1]][,2]-temp[1][[1]][,1])) / 2
#Data Coverage 
m0  = (temp[1][[1]][,1]<=obs[,1]) & (obs[,1]<=temp[1][[1]][,2])
m1 = (temp[1][[1]][,4]<=obs[,2]) & (obs[,2]<=temp[1][[1]][,5])
(sum(m0)+sum(m1))/42


## Coupla
png(filename="Coupla_F.png", width=840, height=480)
temp = Gp
par(mfrow=c(1,2))
plot(obs[,1], col="black", ylim=c(0.0, 20),lwd=2,lty=1, main = "Infected", cex.lab = 1.5, cex.main = 2.0, xlab="Time (Days)", ylab="Number", cex.axis=1.3)#,  xlim=c(37,50))
lines(temp[1][[1]][,1], col="red",lwd=2,lty=2)
lines(temp[1][[1]][,2], col="red",lwd=2,lty=2)
lines(temp[1][[1]][,3], col="red",lwd=2,lty=1)

plot(obs[,2], col="black", ylim=c(0.0, 54.0),lwd=2,lty=1, main = "Recovered", cex.lab = 1.5, cex.main = 2.0, xlab="Time (Days)", ylab="Number", cex.axis=1.3)#, xlim=c(0.7,1.1))
lines(temp[1][[1]][,4], col="red",lwd=2,lty=2)
lines(temp[1][[1]][,5], col="red",lwd=2,lty=2)
lines(temp[1][[1]][,6], col="red",lwd=2,lty=1)
dev.off()
#MSPE
print(temp[c(2,3)])
#ATL
mean(sum(temp[1][[1]][,5]-temp[1][[1]][,4], temp[1][[1]][,2]-temp[1][[1]][,1])) / 2
#Data Coverage 
m0  = (temp[1][[1]][,1]<=obs[,1]) & (obs[,1]<=temp[1][[1]][,2])
m1 = (temp[1][[1]][,4]<=obs[,2]) & (obs[,2]<=temp[1][[1]][,5])
(sum(m0)+sum(m1))/42


