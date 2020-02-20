library(neuralnet)
library(readr)
library(mice)
library(VIM)
library(sqldf)
library(caret)
library(mltools)
library(tensorflow)

kepler <- read_csv("~/Documents/GitHub/ist687-group-kss/cumulative.csv") # get data

# clean data
kepler <- kepler[, -c(4,31, 32)]

md.pattern(kepler)
aggr_plot <- aggr(kepler, col=c('navyblue','red'), numbers=TRUE, sortVars=TRUE, labels=names(data), cex.axis=.7, gap=3, ylab=c("Histogram of missing data","Pattern"))


# 
# tempData <- mice(kepler,m=5,maxit=2,meth='cart',seed=500)
# summary(tempData)
# 
# tmp <- tempData$data
airquality <- airquality

clean <- sqldf("SELECT koi_disposition, koi_pdisposition, koi_period, koi_duration, 
               koi_depth, koi_prad, koi_model_snr, koi_steff, koi_srad 
               FROM kepler WHERE koi_disposition!='CANDIDATE' AND koi_depth IS NOT NULL")

clean <- clean[, -c(2)]

clean_scaled <- preProcess(clean, method = c("center", "scale"))
clean <- predict(clean_scaled, clean)
clean$koi_disposition <- as.factor(clean$koi_disposition)

clean <- cbind(clean[, 2:8], class.ind(clean$koi_disposition))

names(clean)[8:9] <- c("conf", "falsep")

set.seed(123)
smp_size <- floor(0.75 * nrow(clean))
train_ind <- sample(seq_len(nrow(clean)), size = smp_size)

train <- clean[train_ind, ]
test_x <- clean[-train_ind, ]


terms_list <- names(clean)[1:7]
rest_terms <- paste(terms_list, collapse = "+")

f <- as.formula(paste("conf+falsep~", rest_terms))
f

my_model <- neuralnet(f, 
                      data = train, 
                      hidden = c(7,5,3),
                      linear.output = FALSE,
                      lifesign = "full",
                      threshold = 1.0)

summary(my_model)
predicted <- predict(my_model, test_x)

test_x <- cbind(test_x, predicted)

names(test_x)[10:11] <- c("conf_p", "falsep_p")

test_x$conf_p <- round(test_x$conf_p)
test_x$falsep_p <- round(test_x$falsep_p)

accuracy <- (length(which(test_x$conf == test_x$conf_p))/nrow(test_x))*100

plot(my_model)
### this SUCKS, lets do tf 







