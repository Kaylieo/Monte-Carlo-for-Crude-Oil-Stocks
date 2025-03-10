if (!require(remotes)) install.packages("remotes", repos = "https://cran.rstudio.com/")
library(remotes)

# Install an older version of Matrix compatible with R 4.3.3
remotes::install_version("Matrix", version = "1.6-0", repos = "https://cran.r-project.org")

# MASS is usually part of base R, but if needed:
remotes::install_version("MASS", version = "7.3-51.6", repos = "https://cran.r-project.org")

# Install expm version 0.999-2
remotes::install_version("expm", version = "0.999-2", repos = "https://cran.r-project.org")

# Install MSGARCH version 2.50
remotes::install_version("MSGARCH", version = "2.50", repos = "https://cran.r-project.org")