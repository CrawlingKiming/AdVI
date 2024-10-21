######################################################################################
##	Simulation of Rotavirus example
######################################################################################
set.seed(135)
rm(list=ls())

# contact matrix
# simplest assumption: mixing only happens within own age groups (diagonal)
#contact = diag(6)
n.age = 6
# fraction of pop in each age class
frac.pop = c(2,2,2,6,12,36)/(5*12)

# better assumption: mixing is homogeneous (equal contact b/t all age groups)
# this accounts for pop differences b/t groups
contact = matrix(1,nrow=n.age,ncol=n.age)
for(i in 1:6) {
  for(j in 1:i) {
    contact[i,j] = frac.pop[i]/frac.pop[j]
  }
}
#contact = t(contact)
# estimated in the rotavirus paper 
#beta = c(1.30537122, 3.11852393, 0.03206678, 3.24118948, 0.31400672, 0.09410381) 


######################################################################################
##	Construct time-varying mu(t)
######################################################################################
plotweeks = as.Date("2009-12-23")
week.0 = plotweeks[1] - 3.5 # so that t=1 is the midpoint of the first week
mu.0 = 1/(52*5)	# estimated baseline birth rate

# monthly amplitude from 1980-2000, Audrey Dorelien's thesis on birth seasonality in sub-Saharan Africa
amplitude = c(-.17,.01,.03,.25,.12,.03,-.01,.09,.01,.13,-.31,-.17)

# write mu(t) function that returns the month for a given time t
# window version
mu = function(t) {
  mon = as.numeric( months( week.0 + 7*t, abb=TRUE ) )
  mu.0 * ( 1 + amplitude[ mon ] )
}

# mac/linux version
#mu = function(t) { 
#  dummy =  months( week.0 + 7*t, abb=TRUE ) 
#  mon = match(dummy,month.abb)
#  mu.0 * ( 1 + amplitude[ mon ] )
#}

######################################################################################
##	The SIRS Model
######################################################################################

library(deSolve) # integrate ODEs with function lsoda()

# differential equations for the SIR model with seasonality
SIRseason = function(Time, State, Pars) {
  #with(as.list(c(State, Pars)), {
  gamma_s	= Pars[[1]]
  gamma_m	= Pars[[2]]
  delta	= Pars[[3]]
  tau		= Pars[[4]]
  N	  	= Pars[[5]]
  b		= Pars[[6]]
  phi		= Pars[[7]]
  beta1 = Pars[[8]]
  beta2 = Pars[[9]]
  beta3 = Pars[[10]]
  beta4 = Pars[[11]]
  beta5 = Pars[[12]]
  beta6 = Pars[[13]]
  
  beta = c(beta1, beta2, beta3, beta4, beta5, beta6)
  M  = State[1:n.age]
  S = State[(n.age+1):(2*n.age)]
  Is = State[(2*n.age+1):(3*n.age)]
  Im = State[(3*n.age+1):(4*n.age)]
  R = State[(4*n.age+1):(5*n.age-1)]
  #R = State[(4*n.age+1):(5*n.age)]
  
  M_min = c(M[1],M[1:5])
  S_min = c(S[1],S[1:(n.age-1)])
  Is_min = c(Is[1],Is[1:(n.age-1)])
  Im_min = c(Im[1],Im[1:(n.age-1)])
  R_min = c(R[1],R[1:(n.age-1)])
  
  # age movement
  age.minus = c(1/8,1/8,1/8,1/24,1/48,1/144)
  #age.minus = c(1/8,1/8,1/8,1/24,1/48,0)
  age.plus = c(0,1/8,1/8,1/8,1/24,1/48)
  # 6, 6*6 
  beta.t = beta*contact*(1 + b*cos((2*pi*Time - 52*phi)/52)) # n.age x n.age matrix of transmission b/t age groups
  mu = mu(Time)
  Y = (Is+0.5*Im)/(N*frac.pop) # infectious individuals times relative infectiousness (length n.age vector)
  lambda = beta.t%*%Y # force of infection by age group (length n.age vector)
  # relative risk of infection less for mild infections
  lambda_s = 0.24*lambda
  lambda_m = 0.76*lambda
  
  # note: all compartments are now length n.age vectors
  R.last = N - sum(State) # total pop. size is conserved
  # note: need age transitions in R3
  #dM = age.plus*M_min - age.minus*M + mu*N*c(1,rep(0,n.age-1)) - delta*M
  # only allow births into first age group
  # only allow maternal immunity at all until 2nd age group(2-3mo)
  # after that, transfer directly M2 -> S3			
  dM =  mu*N*c(1,0,0,0,0,0) - c(1/8,1/8,0,0,0,0)*M + c(0,1/8,0,0,0,0)*M_min - delta*M
  dS = age.plus*S_min - age.minus*S + delta*M - lambda*S + c(0,0,1/8,0,0,0)*M_min + tau*c(R,R.last)
  dIs = age.plus*Is_min - age.minus*Is + lambda_s*S - gamma_s*Is
  dIm = age.plus*Im_min - age.minus*Im + lambda_m*S - gamma_m*Im
  dR = age.plus[1:(n.age-1)]*R_min[1:(n.age-1)] - age.minus[1:(n.age-1)]*R + gamma_s*Is[1:(n.age-1)] + gamma_m*Im[1:(n.age-1)] - tau*R
  # fix the last age compartment in R to conserve pop size		
  
  return(list(c(dM,dS,dIs,dIm,dR)))
  #})
}


runSIR = function(b,phi, case.t,beta1,beta2,beta3,beta4,beta5,beta6,atol) {
  #print(beta1)
  times = 1:case.t
  N = 324935
  M.0 = N*13/(5*52) # initial number of children <13weeks protected by maternal immunity
  eq = (N-M.0)/5 # compartments besides M; choose starting values equally spread among compartments
  init.vals = c(M=c(M.0,rep(0,n.age-1)),S=rep(eq/n.age,n.age),Is=rep(eq/n.age,n.age),Im=rep(eq/n.age,n.age),R=rep(eq/n.age,n.age-1))
  #init.vals = c(M=c(M.0,rep(0,n.age-1)),S=rep(eq/n.age,n.age),Is=rep(eq/n.age,n.age),Im=rep(eq/n.age,n.age),R=rep(eq/n.age,n.age))
  pars  <- list(	gamma_s	= 1,		# 1/mean infectious period(weeks), severe infection
                 gamma_m	= 2,		# 1/mean infectious period(weeks), mild infection
                 delta	= 1/13,		# 1/mean period of maternal immunity
                 tau	= 	1/52,		# 1/mean period of immunity following infection
                 N	  	= N,		# population(assume constant)
                 b		= b,		# seasonality amplitude
                 phi		= phi,		# seasonality offset (weeks)
                 beta1 = beta1,
                 beta2 = beta2,
                 beta3 = beta3,
                 beta4 = beta4,
                 beta5 = beta5,
                 beta6 = beta6
  )
  ode.out = lsoda(init.vals,times,SIRseason,pars, atol = atol)
  return( ode.out )
}


# fitted posterior mean from rotavirus data (Park et al, 2017)
# they are parmeters of interst in our simulation studies
b = 0.43 
phi = 7.35 
r = 0.90 
rho = 0.027

# they are provided in (Park et al, 2017)
N = 324935
longterm = 5*52 + 118
beta = c(1.30537122, 3.11852393, 0.03206678, 3.24118948, 0.31400672, 0.09410381) 

# run dynamic SIR model
out = runSIR(b,phi,longterm, beta[1], beta[2], beta[3], beta[4], beta[5], beta[6], atol=1e-4)
lastyears = (longterm-118+1):longterm

ode.out = out[lastyears,]
ind.mat = matrix(2:(5*n.age+1),nrow=5,byrow=TRUE)
M.p = ode.out[,ind.mat[1,]]
S.p = ode.out[,ind.mat[2,]];
Is.p = ode.out[,ind.mat[3,]]; Im.p = ode.out[,ind.mat[4,]];
#R.p = ode.out[,ind.mat[5,1:(n.age-1)]];

lambda = matrix(NA,nrow=length(lastyears),ncol=6)
Y = t(t(Is.p+0.5*Im.p)/(N*frac.pop))
for(i in 1:length(lastyears)) {
  beta.t = beta*contact*(1 + b*cos((2*pi*lastyears[i] - 52*phi)/52))
  lambda[i,] = beta.t%*%Y[i,]
}
# expected number of cases 
# S.p : Susceptible 
cases.age = 0.24*lambda*S.p # proportion of cases that develop severe RVGE
cases.report = cases.age*rho # reported cases in each age group
cases.fit = rbind(cases.report[,1],cases.report[,2],cases.report[,3],cases.report[,4],cases.report[,5],cases.report[,6]) #t(cases.report)#rbind(rowSums(cases.report[,1:3]),cases.report[,4],rowSums(cases.report[,5:6]))
#cases.fit = rbind(rowSums(cases.report[,1:3]),cases.report[,4],rowSums(cases.report[,5:6]))
# Simulate cases datasets (3 by 118)
C.age.simul = matrix(0,6,118)
for(i in 1:6){
  for(j in 1:118){
    C.age.simul[i,j] = rnbinom(1,size=0.9,mu=cases.fit[i,j])
  }
}

plot(t(C.age.simul)[,2])

# simulated data
plot(apply(C.age.simul,2,sum),col=2)
lines(apply(cases.report, 1, sum))
#tC = t(C.age.simul)
#save(tC, file = "4paramModel")

######################################################################################
##	inferring model parameters
######################################################################################

# loglikelihood function for the SIR model; NB observation
# note: NB for each age group separately

loglik = function(pars) {
  with(as.list(pars), {
    N = 324935
    if( 	b < 0 || b > 1 ||
         r < 0.5 || r > 1 ||
         rho < 0 || rho > 1 ||
         phi < 2 || phi > (2*pi+2)||
         beta1<0 || beta1 > 4 ||
         beta2<0 || beta2 > 4 ||
         beta3<0 || beta3 > 4 ||
         beta4<0 || beta4 > 4 ||
         beta5<0 || beta5 > 4 ||
         beta6<0 || beta6 > 4 ) { return(-1e10) } # boundaries (uniform priors)
    #longterm = 40*52 + 118 # run for 40 years
    longterm = 20*52 + 118
    #out = runSIR(b,phi,longterm,atol=1e-6)
    out = runSIR(b,phi, longterm, beta1, beta2, beta3, beta4, beta5, beta6,atol=1e-2)
    lastyears = (longterm-118+1):longterm
    
    ode.out = out[lastyears,]
    ind.mat = matrix(2:(5*n.age+1),nrow=5,byrow=TRUE)
    M.p = ode.out[,ind.mat[1,]]
    S.p = ode.out[,ind.mat[2,]];
    Is.p = ode.out[,ind.mat[3,]]; Im.p = ode.out[,ind.mat[4,]];
    #R.p = ode.out[,ind.mat[5,1:(n.age-1)]];
    
    lambda = matrix(NA,nrow=length(lastyears),ncol=n.age)
    Y = t(t(Is.p+0.5*Im.p)/(N*frac.pop))
    for(i in 1:length(lastyears)) {
      beta.t = beta*contact*(1 + b*cos((2*pi*lastyears[i] - 52*phi)/52))
      lambda[i,] = beta.t%*%Y[i,]
    }
    # expected number of cases
    cases.age = 0.24*lambda*S.p # proportion of cases that develop severe RVGE
    cases.report = cases.age*rho # reported cases in each age group
    
    cases.fit = rbind(cases.report[,1],cases.report[,2],cases.report[,3],cases.report[,4],cases.report[,5],cases.report[,6])
    
    sum( dnbinom(C.age.simul,size=r,mu=cases.fit,log=TRUE) )
    })
}
checkpar = c(b=0.43, phi=7.35, r=0.7, 
             rho=0.027, beta1=1.30537122, 
             beta2=3.11852393, beta3=0.3206678, 
             beta4=3.24118948, beta5=0.31400672, beta6=0.09410381)
loglik(checkpar)
######################################################################################

library("readxl")
library(adaptMCMC)
#fpath = paste0("C:/Users/ChoDongKyu/PycharmProjects/Bayesian_Survae/survae_flows-master/experiments/calibration/data/MSIRtest1.csv")

bpath = './TestCSV6/MSIR6_'
time_res = c()
wd = getwd()
init.pars = c(	b = 0.5 , phi = 7 , r = 0.70 , rho = 0.1 ,beta1 = 2,beta2 = 2, beta3 = 2,beta4 = 2,beta5 = 2,beta6 = 2)
N = 50000 #Number of Samples 


for(idx in 1:5){
  
  fpath = paste0(bpath, idx, '.csv', sep='')  
  
  start.time <- Sys.time()
  C.age.simul <- read.csv(file = fpath, header = F)
  C.age.simul = t(C.age.simul)
  out.mcmc <- MCMC(p=loglik, n=50000, init=init.pars, scale=init.pars/4, adapt=TRUE, acc.rate=.3)
  
  end.time <- Sys.time()
  time.taken.fullmcmc <- end.time - start.time
  time_res[idx] <- time.taken.fullmcmc
  
  base_name = paste0('/posteriors_MCMC_50000_model_6_', idx, sep='')
  
  png_name = paste0(wd, base_name, '.png', sep='')
  data_name = paste0(wd, base_name, '.rdata', sep='')

  save(out.mcmc, file=data_name)
  #out.mcmc$samples = out2
  burnt_mcmc = out.mcmc$samples[-(seq(1, 2000)),]
  
  tlist = c(0.43, 7.35, 0.9, 0.027, 1.30537122, 3.11852393, 0.03206678, 3.24118948, 0.31400672, 0.09410381)
  png(png_name,width=4000,height=3000,units='px',res=300)
  
  par(mfrow=c(2,5), cex=1.75, mar=c(4,2,1,1)+0.1) 
  hist(burnt_mcmc[,1], 30, col="#AEAEAE", xlab=expression("w"), main="", ylab="")
  abline(v=tlist[1],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,2], 30, col="#AEAEAE", xlab=expression(phi), main="", ylab="")
  abline(v=tlist[2],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,3], 30, col="#AEAEAE", xlab=expression("r"), main="", ylab="")
  abline(v=tlist[3],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,4], 30, col="#AEAEAE", xlab=expression(rho), main="", ylab="")
  abline(v=tlist[4],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,5], 30, col="#AEAEAE", xlab=expression("b"[1]), main="", ylab="")
  abline(v=tlist[5],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,6], 30, col="#AEAEAE", xlab=expression("b"[2]), main="", ylab="")
  abline(v=tlist[6],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,7], 30, col="#AEAEAE", xlab=expression("b"[3]), main="", ylab="")
  abline(v=tlist[7],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,8], 30, col="#AEAEAE", xlab=expression("b"[4]), main="", ylab="")
  abline(v=tlist[8],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,9], 30, col="#AEAEAE", xlab=expression("b"[5]), main="", ylab="")
  abline(v=tlist[9],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,10], 30, col="#AEAEAE", xlab=expression("b"[6]), main="", ylab="")
  abline(v=tlist[10],col=4,lty=2,lwd=2)
  dev.off()
}


for(idx in 6:10){
  
  fpath = paste0(bpath, idx, '.csv', sep='')  
  
  start.time <- Sys.time()
  C.age.simul <- read.csv(file = fpath, header = F)
  C.age.simul = t(C.age.simul)
  out.mcmc <- MCMC(p=loglik, n=50000, init=init.pars, scale=init.pars/4, adapt=TRUE, acc.rate=.3)
  
  end.time <- Sys.time()
  time.taken.fullmcmc <- end.time - start.time
  time_res[idx] <- time.taken.fullmcmc
  
  base_name = paste0('/posteriors_MCMC_50000_model_6_', idx, sep='')
  
  png_name = paste0(wd, base_name, '.png', sep='')
  data_name = paste0(wd, base_name, '.rdata', sep='')
  
  save(out.mcmc, file=data_name)
  #out.mcmc$samples = out2
  burnt_mcmc = out.mcmc$samples[-(seq(1, 2000)),]
  
  tlist = c(0.43, 7.35, 0.9, 0.027, 1.30537122, 3.11852393, 0.03206678, 3.24118948, 0.31400672, 0.09410381)
  png(png_name,width=4000,height=3000,units='px',res=300)
  
  par(mfrow=c(2,5), cex=1.75, mar=c(4,2,1,1)+0.1) 
  hist(burnt_mcmc[,1], 30, col="#AEAEAE", xlab=expression("w"), main="", ylab="")
  abline(v=tlist[1],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,2], 30, col="#AEAEAE", xlab=expression(phi), main="", ylab="")
  abline(v=tlist[2],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,3], 30, col="#AEAEAE", xlab=expression("r"), main="", ylab="")
  abline(v=tlist[3],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,4], 30, col="#AEAEAE", xlab=expression(rho), main="", ylab="")
  abline(v=tlist[4],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,5], 30, col="#AEAEAE", xlab=expression("b"[1]), main="", ylab="")
  abline(v=tlist[5],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,6], 30, col="#AEAEAE", xlab=expression("b"[2]), main="", ylab="")
  abline(v=tlist[6],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,7], 30, col="#AEAEAE", xlab=expression("b"[3]), main="", ylab="")
  abline(v=tlist[7],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,8], 30, col="#AEAEAE", xlab=expression("b"[4]), main="", ylab="")
  abline(v=tlist[8],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,9], 30, col="#AEAEAE", xlab=expression("b"[5]), main="", ylab="")
  abline(v=tlist[9],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,10], 30, col="#AEAEAE", xlab=expression("b"[6]), main="", ylab="")
  abline(v=tlist[10],col=4,lty=2,lwd=2)
  dev.off()
}


for(idx in 11:15){
  
  fpath = paste0(bpath, idx, '.csv', sep='')  
  
  start.time <- Sys.time()
  C.age.simul <- read.csv(file = fpath, header = F)
  C.age.simul = t(C.age.simul)
  out.mcmc <- MCMC(p=loglik, n=50000, init=init.pars, scale=init.pars/4, adapt=TRUE, acc.rate=.3)
  
  end.time <- Sys.time()
  time.taken.fullmcmc <- end.time - start.time
  time_res[idx] <- time.taken.fullmcmc
  
  base_name = paste0('/posteriors_MCMC_50000_model_6_', idx, sep='')
  
  png_name = paste0(wd, base_name, '.png', sep='')
  data_name = paste0(wd, base_name, '.rdata', sep='')
  
  save(out.mcmc, file=data_name)
  #out.mcmc$samples = out2
  burnt_mcmc = out.mcmc$samples[-(seq(1, 2000)),]
  
  tlist = c(0.43, 7.35, 0.9, 0.027, 1.30537122, 3.11852393, 0.03206678, 3.24118948, 0.31400672, 0.09410381)
  png(png_name,width=4000,height=3000,units='px',res=300)
  
  par(mfrow=c(2,5), cex=1.75, mar=c(4,2,1,1)+0.1) 
  hist(burnt_mcmc[,1], 30, col="#AEAEAE", xlab=expression("w"), main="", ylab="")
  abline(v=tlist[1],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,2], 30, col="#AEAEAE", xlab=expression(phi), main="", ylab="")
  abline(v=tlist[2],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,3], 30, col="#AEAEAE", xlab=expression("r"), main="", ylab="")
  abline(v=tlist[3],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,4], 30, col="#AEAEAE", xlab=expression(rho), main="", ylab="")
  abline(v=tlist[4],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,5], 30, col="#AEAEAE", xlab=expression("b"[1]), main="", ylab="")
  abline(v=tlist[5],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,6], 30, col="#AEAEAE", xlab=expression("b"[2]), main="", ylab="")
  abline(v=tlist[6],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,7], 30, col="#AEAEAE", xlab=expression("b"[3]), main="", ylab="")
  abline(v=tlist[7],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,8], 30, col="#AEAEAE", xlab=expression("b"[4]), main="", ylab="")
  abline(v=tlist[8],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,9], 30, col="#AEAEAE", xlab=expression("b"[5]), main="", ylab="")
  abline(v=tlist[9],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,10], 30, col="#AEAEAE", xlab=expression("b"[6]), main="", ylab="")
  abline(v=tlist[10],col=4,lty=2,lwd=2)
  dev.off()
}


for(idx in 16:20){
  
  fpath = paste0(bpath, idx, '.csv', sep='')  
  
  start.time <- Sys.time()
  C.age.simul <- read.csv(file = fpath, header = F)
  C.age.simul = t(C.age.simul)
  out.mcmc <- MCMC(p=loglik, n=50000, init=init.pars, scale=init.pars/4, adapt=TRUE, acc.rate=.3)
  
  end.time <- Sys.time()
  time.taken.fullmcmc <- end.time - start.time
  time_res[idx] <- time.taken.fullmcmc
  
  base_name = paste0('/posteriors_MCMC_50000_model_6_', idx, sep='')
  
  png_name = paste0(wd, base_name, '.png', sep='')
  data_name = paste0(wd, base_name, '.rdata', sep='')
  
  save(out.mcmc, file=data_name)
  #out.mcmc$samples = out2
  burnt_mcmc = out.mcmc$samples[-(seq(1, 2000)),]
  
  tlist = c(0.43, 7.35, 0.9, 0.027, 1.30537122, 3.11852393, 0.03206678, 3.24118948, 0.31400672, 0.09410381)
  png(png_name,width=4000,height=3000,units='px',res=300)
  
  par(mfrow=c(2,5), cex=1.75, mar=c(4,2,1,1)+0.1) 
  hist(burnt_mcmc[,1], 30, col="#AEAEAE", xlab=expression("w"), main="", ylab="")
  abline(v=tlist[1],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,2], 30, col="#AEAEAE", xlab=expression(phi), main="", ylab="")
  abline(v=tlist[2],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,3], 30, col="#AEAEAE", xlab=expression("r"), main="", ylab="")
  abline(v=tlist[3],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,4], 30, col="#AEAEAE", xlab=expression(rho), main="", ylab="")
  abline(v=tlist[4],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,5], 30, col="#AEAEAE", xlab=expression("b"[1]), main="", ylab="")
  abline(v=tlist[5],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,6], 30, col="#AEAEAE", xlab=expression("b"[2]), main="", ylab="")
  abline(v=tlist[6],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,7], 30, col="#AEAEAE", xlab=expression("b"[3]), main="", ylab="")
  abline(v=tlist[7],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,8], 30, col="#AEAEAE", xlab=expression("b"[4]), main="", ylab="")
  abline(v=tlist[8],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,9], 30, col="#AEAEAE", xlab=expression("b"[5]), main="", ylab="")
  abline(v=tlist[9],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,10], 30, col="#AEAEAE", xlab=expression("b"[6]), main="", ylab="")
  abline(v=tlist[10],col=4,lty=2,lwd=2)
  dev.off()
}


for(idx in 21:25){
  
  fpath = paste0(bpath, idx, '.csv', sep='')  
  
  start.time <- Sys.time()
  C.age.simul <- read.csv(file = fpath, header = F)
  C.age.simul = t(C.age.simul)
  out.mcmc <- MCMC(p=loglik, n=50000, init=init.pars, scale=init.pars/4, adapt=TRUE, acc.rate=.3)
  
  end.time <- Sys.time()
  time.taken.fullmcmc <- end.time - start.time
  time_res[idx] <- time.taken.fullmcmc
  
  base_name = paste0('/posteriors_MCMC_50000_model_6_', idx, sep='')
  
  png_name = paste0(wd, base_name, '.png', sep='')
  data_name = paste0(wd, base_name, '.rdata', sep='')
  
  save(out.mcmc, file=data_name)
  #out.mcmc$samples = out2
  burnt_mcmc = out.mcmc$samples[-(seq(1, 2000)),]
  
  tlist = c(0.43, 7.35, 0.9, 0.027, 1.30537122, 3.11852393, 0.03206678, 3.24118948, 0.31400672, 0.09410381)
  png(png_name,width=4000,height=3000,units='px',res=300)
  
  par(mfrow=c(2,5), cex=1.75, mar=c(4,2,1,1)+0.1) 
  hist(burnt_mcmc[,1], 30, col="#AEAEAE", xlab=expression("w"), main="", ylab="")
  abline(v=tlist[1],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,2], 30, col="#AEAEAE", xlab=expression(phi), main="", ylab="")
  abline(v=tlist[2],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,3], 30, col="#AEAEAE", xlab=expression("r"), main="", ylab="")
  abline(v=tlist[3],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,4], 30, col="#AEAEAE", xlab=expression(rho), main="", ylab="")
  abline(v=tlist[4],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,5], 30, col="#AEAEAE", xlab=expression("b"[1]), main="", ylab="")
  abline(v=tlist[5],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,6], 30, col="#AEAEAE", xlab=expression("b"[2]), main="", ylab="")
  abline(v=tlist[6],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,7], 30, col="#AEAEAE", xlab=expression("b"[3]), main="", ylab="")
  abline(v=tlist[7],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,8], 30, col="#AEAEAE", xlab=expression("b"[4]), main="", ylab="")
  abline(v=tlist[8],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,9], 30, col="#AEAEAE", xlab=expression("b"[5]), main="", ylab="")
  abline(v=tlist[9],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,10], 30, col="#AEAEAE", xlab=expression("b"[6]), main="", ylab="")
  abline(v=tlist[10],col=4,lty=2,lwd=2)
  dev.off()
}


for(idx in 26:30){
  
  fpath = paste0(bpath, idx, '.csv', sep='')  
  
  start.time <- Sys.time()
  C.age.simul <- read.csv(file = fpath, header = F)
  C.age.simul = t(C.age.simul)
  out.mcmc <- MCMC(p=loglik, n=50000, init=init.pars, scale=init.pars/4, adapt=TRUE, acc.rate=.3)
  
  end.time <- Sys.time()
  time.taken.fullmcmc <- end.time - start.time
  time_res[idx] <- time.taken.fullmcmc
  
  base_name = paste0('/posteriors_MCMC_50000_model_6_', idx, sep='')
  
  png_name = paste0(wd, base_name, '.png', sep='')
  data_name = paste0(wd, base_name, '.rdata', sep='')
  
  save(out.mcmc, file=data_name)
  #out.mcmc$samples = out2
  burnt_mcmc = out.mcmc$samples[-(seq(1, 2000)),]
  
  tlist = c(0.43, 7.35, 0.9, 0.027, 1.30537122, 3.11852393, 0.03206678, 3.24118948, 0.31400672, 0.09410381)
  png(png_name,width=4000,height=3000,units='px',res=300)
  
  par(mfrow=c(2,5), cex=1.75, mar=c(4,2,1,1)+0.1) 
  hist(burnt_mcmc[,1], 30, col="#AEAEAE", xlab=expression("w"), main="", ylab="")
  abline(v=tlist[1],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,2], 30, col="#AEAEAE", xlab=expression(phi), main="", ylab="")
  abline(v=tlist[2],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,3], 30, col="#AEAEAE", xlab=expression("r"), main="", ylab="")
  abline(v=tlist[3],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,4], 30, col="#AEAEAE", xlab=expression(rho), main="", ylab="")
  abline(v=tlist[4],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,5], 30, col="#AEAEAE", xlab=expression("b"[1]), main="", ylab="")
  abline(v=tlist[5],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,6], 30, col="#AEAEAE", xlab=expression("b"[2]), main="", ylab="")
  abline(v=tlist[6],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,7], 30, col="#AEAEAE", xlab=expression("b"[3]), main="", ylab="")
  abline(v=tlist[7],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,8], 30, col="#AEAEAE", xlab=expression("b"[4]), main="", ylab="")
  abline(v=tlist[8],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,9], 30, col="#AEAEAE", xlab=expression("b"[5]), main="", ylab="")
  abline(v=tlist[9],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,10], 30, col="#AEAEAE", xlab=expression("b"[6]), main="", ylab="")
  abline(v=tlist[10],col=4,lty=2,lwd=2)
  dev.off()
}


for(idx in 31:35){
  
  fpath = paste0(bpath, idx, '.csv', sep='')  
  
  start.time <- Sys.time()
  C.age.simul <- read.csv(file = fpath, header = F)
  C.age.simul = t(C.age.simul)
  out.mcmc <- MCMC(p=loglik, n=50000, init=init.pars, scale=init.pars/4, adapt=TRUE, acc.rate=.3)
  
  end.time <- Sys.time()
  time.taken.fullmcmc <- end.time - start.time
  time_res[idx] <- time.taken.fullmcmc
  
  base_name = paste0('/posteriors_MCMC_50000_model_6_', idx, sep='')
  
  png_name = paste0(wd, base_name, '.png', sep='')
  data_name = paste0(wd, base_name, '.rdata', sep='')
  
  save(out.mcmc, file=data_name)
  #out.mcmc$samples = out2
  burnt_mcmc = out.mcmc$samples[-(seq(1, 2000)),]
  
  tlist = c(0.43, 7.35, 0.9, 0.027, 1.30537122, 3.11852393, 0.03206678, 3.24118948, 0.31400672, 0.09410381)
  png(png_name,width=4000,height=3000,units='px',res=300)
  
  par(mfrow=c(2,5), cex=1.75, mar=c(4,2,1,1)+0.1) 
  hist(burnt_mcmc[,1], 30, col="#AEAEAE", xlab=expression("w"), main="", ylab="")
  abline(v=tlist[1],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,2], 30, col="#AEAEAE", xlab=expression(phi), main="", ylab="")
  abline(v=tlist[2],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,3], 30, col="#AEAEAE", xlab=expression("r"), main="", ylab="")
  abline(v=tlist[3],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,4], 30, col="#AEAEAE", xlab=expression(rho), main="", ylab="")
  abline(v=tlist[4],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,5], 30, col="#AEAEAE", xlab=expression("b"[1]), main="", ylab="")
  abline(v=tlist[5],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,6], 30, col="#AEAEAE", xlab=expression("b"[2]), main="", ylab="")
  abline(v=tlist[6],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,7], 30, col="#AEAEAE", xlab=expression("b"[3]), main="", ylab="")
  abline(v=tlist[7],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,8], 30, col="#AEAEAE", xlab=expression("b"[4]), main="", ylab="")
  abline(v=tlist[8],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,9], 30, col="#AEAEAE", xlab=expression("b"[5]), main="", ylab="")
  abline(v=tlist[9],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,10], 30, col="#AEAEAE", xlab=expression("b"[6]), main="", ylab="")
  abline(v=tlist[10],col=4,lty=2,lwd=2)
  dev.off()
}


for(idx in 36:40){
  
  fpath = paste0(bpath, idx, '.csv', sep='')  
  
  start.time <- Sys.time()
  C.age.simul <- read.csv(file = fpath, header = F)
  C.age.simul = t(C.age.simul)
  out.mcmc <- MCMC(p=loglik, n=50000, init=init.pars, scale=init.pars/4, adapt=TRUE, acc.rate=.3)
  
  end.time <- Sys.time()
  time.taken.fullmcmc <- end.time - start.time
  time_res[idx] <- time.taken.fullmcmc
  
  base_name = paste0('/posteriors_MCMC_50000_model_6_', idx, sep='')
  
  png_name = paste0(wd, base_name, '.png', sep='')
  data_name = paste0(wd, base_name, '.rdata', sep='')
  
  save(out.mcmc, file=data_name)
  #out.mcmc$samples = out2
  burnt_mcmc = out.mcmc$samples[-(seq(1, 2000)),]
  
  tlist = c(0.43, 7.35, 0.9, 0.027, 1.30537122, 3.11852393, 0.03206678, 3.24118948, 0.31400672, 0.09410381)
  png(png_name,width=4000,height=3000,units='px',res=300)
  
  par(mfrow=c(2,5), cex=1.75, mar=c(4,2,1,1)+0.1) 
  hist(burnt_mcmc[,1], 30, col="#AEAEAE", xlab=expression("w"), main="", ylab="")
  abline(v=tlist[1],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,2], 30, col="#AEAEAE", xlab=expression(phi), main="", ylab="")
  abline(v=tlist[2],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,3], 30, col="#AEAEAE", xlab=expression("r"), main="", ylab="")
  abline(v=tlist[3],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,4], 30, col="#AEAEAE", xlab=expression(rho), main="", ylab="")
  abline(v=tlist[4],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,5], 30, col="#AEAEAE", xlab=expression("b"[1]), main="", ylab="")
  abline(v=tlist[5],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,6], 30, col="#AEAEAE", xlab=expression("b"[2]), main="", ylab="")
  abline(v=tlist[6],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,7], 30, col="#AEAEAE", xlab=expression("b"[3]), main="", ylab="")
  abline(v=tlist[7],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,8], 30, col="#AEAEAE", xlab=expression("b"[4]), main="", ylab="")
  abline(v=tlist[8],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,9], 30, col="#AEAEAE", xlab=expression("b"[5]), main="", ylab="")
  abline(v=tlist[9],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,10], 30, col="#AEAEAE", xlab=expression("b"[6]), main="", ylab="")
  abline(v=tlist[10],col=4,lty=2,lwd=2)
  dev.off()
}


for(idx in 41:45){
  
  fpath = paste0(bpath, idx, '.csv', sep='')  
  
  start.time <- Sys.time()
  C.age.simul <- read.csv(file = fpath, header = F)
  C.age.simul = t(C.age.simul)
  out.mcmc <- MCMC(p=loglik, n=50000, init=init.pars, scale=init.pars/4, adapt=TRUE, acc.rate=.3)
  
  end.time <- Sys.time()
  time.taken.fullmcmc <- end.time - start.time
  time_res[idx] <- time.taken.fullmcmc
  
  base_name = paste0('/posteriors_MCMC_50000_model_6_', idx, sep='')
  
  png_name = paste0(wd, base_name, '.png', sep='')
  data_name = paste0(wd, base_name, '.rdata', sep='')
  
  save(out.mcmc, file=data_name)
  #out.mcmc$samples = out2
  burnt_mcmc = out.mcmc$samples[-(seq(1, 2000)),]
  
  tlist = c(0.43, 7.35, 0.9, 0.027, 1.30537122, 3.11852393, 0.03206678, 3.24118948, 0.31400672, 0.09410381)
  png(png_name,width=4000,height=3000,units='px',res=300)
  
  par(mfrow=c(2,5), cex=1.75, mar=c(4,2,1,1)+0.1) 
  hist(burnt_mcmc[,1], 30, col="#AEAEAE", xlab=expression("w"), main="", ylab="")
  abline(v=tlist[1],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,2], 30, col="#AEAEAE", xlab=expression(phi), main="", ylab="")
  abline(v=tlist[2],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,3], 30, col="#AEAEAE", xlab=expression("r"), main="", ylab="")
  abline(v=tlist[3],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,4], 30, col="#AEAEAE", xlab=expression(rho), main="", ylab="")
  abline(v=tlist[4],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,5], 30, col="#AEAEAE", xlab=expression("b"[1]), main="", ylab="")
  abline(v=tlist[5],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,6], 30, col="#AEAEAE", xlab=expression("b"[2]), main="", ylab="")
  abline(v=tlist[6],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,7], 30, col="#AEAEAE", xlab=expression("b"[3]), main="", ylab="")
  abline(v=tlist[7],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,8], 30, col="#AEAEAE", xlab=expression("b"[4]), main="", ylab="")
  abline(v=tlist[8],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,9], 30, col="#AEAEAE", xlab=expression("b"[5]), main="", ylab="")
  abline(v=tlist[9],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,10], 30, col="#AEAEAE", xlab=expression("b"[6]), main="", ylab="")
  abline(v=tlist[10],col=4,lty=2,lwd=2)
  dev.off()
}


for(idx in 46:50){
  
  fpath = paste0(bpath, idx, '.csv', sep='')  
  
  start.time <- Sys.time()
  C.age.simul <- read.csv(file = fpath, header = F)
  C.age.simul = t(C.age.simul)
  out.mcmc <- MCMC(p=loglik, n=50000, init=init.pars, scale=init.pars/4, adapt=TRUE, acc.rate=.3)
  
  end.time <- Sys.time()
  time.taken.fullmcmc <- end.time - start.time
  time_res[idx] <- time.taken.fullmcmc
  
  base_name = paste0('/posteriors_MCMC_50000_model_6_', idx, sep='')
  
  png_name = paste0(wd, base_name, '.png', sep='')
  data_name = paste0(wd, base_name, '.rdata', sep='')
  
  save(out.mcmc, file=data_name)
  #out.mcmc$samples = out2
  burnt_mcmc = out.mcmc$samples[-(seq(1, 2000)),]
  
  tlist = c(0.43, 7.35, 0.9, 0.027, 1.30537122, 3.11852393, 0.03206678, 3.24118948, 0.31400672, 0.09410381)
  png(png_name,width=4000,height=3000,units='px',res=300)
  
  par(mfrow=c(2,5), cex=1.75, mar=c(4,2,1,1)+0.1) 
  hist(burnt_mcmc[,1], 30, col="#AEAEAE", xlab=expression("w"), main="", ylab="")
  abline(v=tlist[1],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,2], 30, col="#AEAEAE", xlab=expression(phi), main="", ylab="")
  abline(v=tlist[2],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,3], 30, col="#AEAEAE", xlab=expression("r"), main="", ylab="")
  abline(v=tlist[3],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,4], 30, col="#AEAEAE", xlab=expression(rho), main="", ylab="")
  abline(v=tlist[4],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,5], 30, col="#AEAEAE", xlab=expression("b"[1]), main="", ylab="")
  abline(v=tlist[5],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,6], 30, col="#AEAEAE", xlab=expression("b"[2]), main="", ylab="")
  abline(v=tlist[6],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,7], 30, col="#AEAEAE", xlab=expression("b"[3]), main="", ylab="")
  abline(v=tlist[7],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,8], 30, col="#AEAEAE", xlab=expression("b"[4]), main="", ylab="")
  abline(v=tlist[8],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,9], 30, col="#AEAEAE", xlab=expression("b"[5]), main="", ylab="")
  abline(v=tlist[9],col=4,lty=2,lwd=2)
  hist(burnt_mcmc[,10], 30, col="#AEAEAE", xlab=expression("b"[6]), main="", ylab="")
  abline(v=tlist[10],col=4,lty=2,lwd=2)
  dev.off()
}
