#===============================================#
# Simulating GPP vom PREles with available data #
#===============================================#

#devtools::install_github('MikkoPeltoniemi/Rpreles')

# load packages
library(Rpreles)
library(BayesianTools)

# load data
load("data/soro_clim.Rdata")
load("data/soro_gpp.Rdata")
# load default parameter values
load("rdata/ParameterRangesPreles.RData")

parind <- c(5:10, 14:16) # Indexes for PRELES parameters

#==================#
# Calibrate PREles #
#==================#

#1- Likelihood function
likelihood <- function(pValues){
  p <- par$def
  p[parind] <- pValues # new parameter values
  predicted<- PRELES(DOY=X$DOY,PAR=X$PAR,TAir=X$TAir,VPD=X$VPD,Precip=X$Precip,
                     CO2=X$CO2,fAPAR=X$fAPAR,p=p[1:30])
  diff_GPP <- predicted$GPP-y$GPP
  # mäkälä
  llvalues <- sum(dnorm(predicted$GPP, mean = y$GPP, sd = p[31], log=T)) ###   llvalues <- sum(dnorm(diff_GPP, sd = p[31], log=T))
  #llvalues <- sum(dexp(abs(diff_GPP),rate = 1/(p[31]+p[32]*predicted$GPP),log=T))
  return(llvalues)
}

#2- Prior
prior <- createUniformPrior(lower = par$min[parind], upper = par$max[parind])

#=Bayesian set up=#

BSpreles <- createBayesianSetup(likelihood, prior, best = par$def[parind], 
                                names = par$name[parind], parallel = F)

bssetup <- checkBayesianSetup(BSpreles)

#=Run the MCMC with three chains=#

settings <- data.frame(iterations = 1e5, nrChains = 3)
chainDE <- runMCMC(BSpreles, sampler="DEzs", settings = settings)
par.opt<-MAP(chainDE) #gets the optimized maximum value for the parameters

# Check convergence:
tracePlot(chainDE, parametersOnly = TRUE, start = 1, whichParameters = 1:4)
tracePlot(chainDE, parametersOnly = TRUE, start = 1, whichParameters = 5:9)

marginalPlot(chainDE, scale = T, best = T, start = 5000)
correlationPlot(chainDE, parametersOnly = TRUE, start = 2000)

# save calibrated parameters
par$calib = par$def
par$calib[parind] = par.opt$parametersMAP
save(par, file = "rdata/CalibratedParametersSoro.Rdata")
write.table(par, file="data/params_calibrated", sep = ";",row.names = FALSE)

#======================#
# Simulate from PREles #
#======================#

output <- PRELES(TAir = X$TAir, PAR = X$PAR, VPD = X$VPD, Precip = X$Precip, fAPAR = X$fAPAR, CO2 = X$CO2, p = par$calib[1:30], returncols = c("GPP", "SW", "ET"))

y_preles = as.data.frame(do.call(cbind, output))

save(y_preles, file="data/soro_preles_gpp.Rdata")
write.table(y_preles, file="data/soro_preles_gpp", sep = ";",row.names = FALSE)
