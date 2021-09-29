setwd("C:/Users/ruben/Documents/Practicals/R-Practicals/R_practice/Decision_Trees")
#Load the datasets 
library(ggplot2)
library(dplyr)
library(rpart)
library(rpart.plot)
library(caret)
library(randomForest)
library(cowplot)
library(GGally)
library(corrplot)
library(plyr)
library(RColorBrewer)
getwd()


#setwd("~/Oct-03/Decision_Tree_10_25")
rm(list = ls())

# Load the dataset
income <- read.csv("Income_Dataset.csv")
View(income)
str(income)
summary(income)

##Exploratory data analysis
#Plot the categorical variables with the income to determine  which factor affects the income
bar_theme1<- theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5), 
                   legend.position="right") #Assigning a variable for theme customization of the bar chart
plot_grid(ggplot(income,aes(x = workclass,fill = income)) + geom_bar() + bar_theme1,
         ggplot(income,aes(x = education,fill = income)) + geom_bar() + bar_theme1,
         ggplot(income,aes(x = factor(education.num),fill = income)) + geom_bar() + bar_theme1,
         ggplot(income,aes(x = marital.status,fill = income)) + geom_bar() + bar_theme1,
         ggplot(income,aes(x = occupation,fill = income)) + geom_bar() + bar_theme1,
         ggplot(income,aes(x = relationship,fill = income)) + geom_bar() + bar_theme1,
         ggplot(income,aes(x = race,fill = income)) + geom_bar() + bar_theme1,
         ggplot(income,aes(x = sex,fill = income)) + geom_bar() + bar_theme1,
         ggplot(income,aes(x = native.country,fill = income)) + geom_bar() + bar_theme1,
         align = "h")

#Histogram and boxplot for the numerical variables
plot_grid(ggplot(income,aes(x=age,fill = income)) + geom_histogram() + bar_theme1,
          ggplot(income,aes(x=fnlwgt,fill = income)) + geom_histogram() + bar_theme1,
          ggplot(income,aes(x=capital.gain,fill = income)) + geom_histogram() + bar_theme1,
          ggplot(income,aes(x=capital.loss,fill = income)) + geom_histogram() + bar_theme1,
          ggplot(income,aes(x=hours.per.week,fill = income)) + geom_histogram() + bar_theme1,
          ggplot(income,aes(x=age)) + geom_boxplot() + bar_theme1,
          ggplot(income,aes(x=fnlwgt)) + geom_boxplot() + bar_theme1,
          ggplot(income,aes(x=capital.gain)) + geom_boxplot() + bar_theme1,
          ggplot(income,aes(x=capital.loss)) + geom_boxplot() + bar_theme1)

#correlation between the continuous variables
ggpairs(income[,c("age","fnlwgt","capital.gain","capital.loss","hours.per.week","income")])

##Data cleansing 
data_filter <- subset(income,occupation == "?")
income<- income[!income$workclass %in% data_filter$workclass,]#Dropping the records with missing information

#Changing the income column to a form fit for modelling
#str(income)
#income$income <- ifelse(income$income == "<=50K",0,1)

#Some more data visualizations and understanding correlations between numeric variables using corrplot
corr = cor(income[,c("age","fnlwgt","capital.gain","capital.loss","hours.per.week","income")])
corrplot(corr = corr, method = "color", outline = T, cl.pos = 'n', rect.col = "black",  tl.col = "indianred4", addCoef.col = "black", number.digits = 2, number.cex = 0.60, tl.cex = 0.7, cl.cex = 1, col = colorRampPalette(c("green4","white","red"))(100))

#Creating a character dataframe and using chi-sqr test to check correlations betweeen categorical features and target variable
income_chr <- income[,-c(1,3,11,12,13)]
income_fact <- data.frame(sapply(income_chr, function(x) factor(x)))
chisq.test(table(income_fact$workclass,income_chr$income))#Workclass is correlated with income 
chisq.test(table(income_fact$education,income_fact$income)) # education is correlated with income  
chisq.test(table(income_fact$education.num,income_fact$income)) # education.num is correlated with income  
chisq.test(table(income_fact$marital.status,income_fact$income)) # marital status is correlated with income  
chisq.test(table(income_fact$occupation,income_fact$income)) # occupation is correlated with income  
chisq.test(table(income_fact$relationship,income_fact$income)) # relationship is correlated with income  
chisq.test(table(income_fact$race,income_fact$income)) # race is correlated with income  

##Data Preparation
#Check for missing values
sapply(income, function(x) sum(is.na(x))) #no missing values were observed

#Outlier detection
str(income)
iqr1<-IQR(income$age)
quant1 <- quantile(income$age)
ll1 <- round(quant1[2]-iqr1*1.5)   
ul1 <- round(quant1[4]+iqr1*1.5)
View(subset(income,age<ll1|age>ul1))   #only for age, the number of records above and below the limits are quite few. Hence we drop these records  
income <- subset(income,age>ll1&age<ul1)#Drop the values for feature 'age' which are beyond their limits
#for the other features, we use outlier treatment and missing value imputation
sapply(income[,c('fnlwgt','capital.gain','capital.loss')], function(x) quantile(x,seq(0,1,0.01),na.rm = T))

##Data modelling
set.seed(123)
split.indices <- sample(nrow(income),nrow(income)*0.7,replace = F)
train<- income[split.indices,]
test<- income[-split.indices,]

#Create the decison tree model
#Classification model
tree.model<- rpart(income~.,data = train,method = "class")
prp(tree.model)
#Testing the model on the test set
tree.predict <- predict(tree.model,test,type = "class")
confusionMatrix(tree.predict, as.factor(test$income), positive = ">50K")

#Tune the hypermarameters
tree.model_tuned <- rpart(income~.,data = train, method = "class", control = rpart.control(minbucket =250,minsplit = 250,cp = 0.02))
prp(tree.model_tuned)
#Testing the model on the test set
tree.predict_tuned <- predict(tree.model_tuned,test,type="class")
confusionMatrix(tree.predict_tuned,as.factor(test$income),positive = ">50K")

#Build a random forest 
memory.limit(56000)
set.seed(234)

data.rf <- randomForest(as.factor(income)~.,data = train,ntree = 100,mtry = 5,do.trace = FALSE,na.action = na.omit)  #the target value should be a factor
data.rf
testPred <- predict(data.rf,test, type = "class")
confusionMatrix(testPred,as.factor(test$income),positive = ">50K")
matr <- table(test$income,testPred)
coul<- colorRampPalette(brewer.pal(8,"PiYG"))(25)
heatmap(matr, scale = "none",col = coul)