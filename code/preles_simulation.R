#===============================================#
# Simulating GPP vom PREles with available data #
#===============================================#

#devtools::install_github('MikkoPeltoniemi/Rpreles')

# load packages
library(Rpreles)
library(BayesianTools)

# load data
load("~/Documents/Projects/Intern_MPI_Jena/data/soro_clim.Rdata")
# load default parameter values
load("~/Documents/Projects/Intern_MPI_Jena/rdata/ParameterRangesPreles.RData")


#==================#
# Calibrate PREles #
#==================#

#1- Likelihood function
likelihood <- function(pValues){
  p <- par$def
  p[parind] <- pValues # new parameter values
  predicted<- PRELES(DOY=profound_in$DOY,PAR=profound_in$PAR,TAir=profound_in$TAir,VPD=profound_in$VPD,Precip=profound_in$Precip,
                     CO2=profound_in$CO2,fAPAR=profound_in$fAPAR,p=p[1:30])
  diff_GPP <- predicted$GPP-profound_out
  # mäkälä
  llvalues <- sum(dnorm(predicted$GPP, mean = profound_out, sd = p[31], log=T)) ###   llvalues <- sum(dnorm(diff_GPP, sd = p[31], log=T))
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

settings <- data.frame(iterations = 1e5, optimize=F, nrChains = 3)
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
save(par, file = "~/Sc_Master/Masterthesis/Project/DomAdapt/Rdata/CalibratedParametersHytProf.Rdata")

#======================#
# Simulate from PREles #
#======================#

output <- PRELES(TAir = X$TAir, PAR = X$PAR, VPD = X$VPD, Precip = X$Precip, 
                 fAPAR = X$fAPAR, CO2 = X$CO2,
                 returncols = c("GPP", "SW", "ET"))

y_preles = as.data.frame(do.call(cbind, output))

save(y_preles, file="~/Documents/Projects/Intern_MPI_Jena/data/preles_sims.Rdata")
write.table(y_preles, file="~/Documents/Projects/Intern_MPI_Jena/data/preles_sims", sep = ";",row.names = FALSE)
