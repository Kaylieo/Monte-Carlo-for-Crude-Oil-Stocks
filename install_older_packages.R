if (!require(remotes)) install.packages("remotes", repos = "https://cran.rstudio.com/")
library(remotes)

# Install MSGARCH version 2.50
remotes::install_version("MSGARCH", version = "2.50", repos = "https://cran.r-project.org")