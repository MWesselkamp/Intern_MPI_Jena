#===============================================#
# Simulating GPP vom PREles with available data #
#===============================================#

#devtools::install_github('MikkoPeltoniemi/Rpreles')

# load packages
library(Rpreles)

# load data
load("~/Documents/Projects/Intern_MPI_Jena/data/soro_clim.Rdata")

output <- PRELES(TAir = X$TAir, PAR = X$PAR, VPD = X$VPD, Precip = X$Precip, 
                 fAPAR = X$fAPAR, CO2 = X$CO2,
                 returncols = c("GPP", "SW", "ET"))

y_preles = as.data.frame(do.call(cbind, output))

save(y_preles, file="~/Documents/Projects/Intern_MPI_Jena/data/preles_sims.Rdata")
write.table(y_preles, file="~/Documents/Projects/Intern_MPI_Jena/data/preles_sims", sep = ";",row.names = FALSE)
