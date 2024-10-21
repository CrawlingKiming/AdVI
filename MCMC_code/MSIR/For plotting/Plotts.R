BTAT = read.csv(file="BTAT_num27.csv", header = FALSE)
BFAF = read.csv(file="BFAF_num27.csv", header = FALSE)
Coupla = read.csv(file="copula_num27.csv", header=FALSE)
MCMC = read.csv(file="SampleCSV27.csv", header=TRUE)
MCMC[,c(3,4)] = MCMC[,c(4,3)]
library(readxl)


pMSIR = function(mat, name){
  temp = mat 
  tlist = c(0.43, 7.35, 0.9, 0.027, 1.30537122, 3.11852393, 0.03206678, 3.24118948, 0.31400672, 0.09410381)
  png(name,width=4500,height=3200,units='px',res=300)
  par(mfrow=c(3,4), cex=1.75, mar=c(4,2,1,1)+0.1) 
  plot(density(temp[,1], adjust=1.5), col="black",main=expression("w"), cex.lab = 1.3, cex.main = 1.6, ylab="", cex.axis=1.4, xlab="", xlim=c(0.2, 0.8))
  abline(v=tlist[1],col='red',lty=2,lwd=2)
  plot(density(temp[,2], adjust=1.5), col="black", main=expression(phi), cex.lab = 1.3, cex.main = 1.6, ylab="", cex.axis=1.4, xlab="", xlim=c(6, 7.9))
  abline(v=tlist[2],col='red',lty=2,lwd=2)
  plot(density(temp[,4], adjust=1.5, to=1.0), col="black", main=expression("r"), cex.lab = 1.3, cex.main = 1.6, ylab="", cex.axis=1.4, xlab="", xlim=c(0.6, 1.0))
  abline(v=tlist[3],col='red',lty=2,lwd=2)
  plot(density(temp[,3], adjust=1.5), col="black", main=expression(rho), cex.lab = 1.3, cex.main = 1.6, ylab="", cex.axis=1.4, xlab="", xlim=c(0.02, 0.05))
  abline(v=tlist[4],col='red',lty=2,lwd=2)
  plot(density(temp[,5], adjust=1.5), col="black", main=expression(beta[01]), cex.lab = 1.3, cex.main = 1.6, ylab="", cex.axis=1.4, xlab="", xlim=c(0.0, 3.0))
  abline(v=tlist[5],col='red',lty=2,lwd=2)
  plot(density(temp[,6], adjust=1.5), col="black", main=expression(beta[02]), cex.lab = 1.3, cex.main = 1.6, ylab="", cex.axis=1.4, xlab="", xlim=c(1.5, 4.0))
  abline(v=tlist[6],col='red',lty=2,lwd=2)
  plot(density(temp[,7], adjust=1.5, from=0.0), col="black", main=expression(beta[03]), cex.lab = 1.3, cex.main = 1.6, ylab="", cex.axis=1.4, xlab="", xlim=c(0.0, 1.5))
  abline(v=tlist[7],col='red',lty=2,lwd=2)
  plot(density(temp[,8], adjust=1.5), col="black", main=expression(beta[04]), cex.lab = 1.3, cex.main = 1.6, ylab="", cex.axis=1.4, xlab="", xlim=c(1.5, 4.0))
  abline(v=tlist[8],col='red',lty=2,lwd=2)
  plot(density(temp[,9], adjust=1.5, from=0.0), col="black", main=expression(beta[05]), cex.lab = 1.3, cex.main =1.6, ylab="", cex.axis=1.4, xlab="", xlim=c(0.0, 1.5))
  abline(v=tlist[9],col='red',lty=2,lwd=2)
  plot(density(temp[,10], adjust=1.5, from=0.0), col="black", main=expression(beta[06]), cex.lab = 1.3, cex.main = 1.6, ylab="", cex.axis=1.4, xlab="", xlim=c(0.0, 1.5))
  abline(v=tlist[10],col='red',lty=2,lwd=2)
  dev.off()
}


pMSIR(BTAT, "BTAT_MSIR.png")
pMSIR(Coupla, "Copula_MSIR.png")
pMSIR(BFAF, "BFAF_MSIR.png")
pMSIR(MCMC, "MCMC_MSIR.png")



