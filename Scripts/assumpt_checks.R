rm(list=ls())
library(nlme);
library(lmerTest);
#library(lattice); 
#library(dplyr)
#library(GLMcat)
options(contrasts=c("contr.sum","contr.poly"))
require(nlme)  
library(performance)
library(parameters)

### READ DATA
dc<-read.csv("C:/Users/Margherita/Documents/GitHub//SOABB1/Scripts/N72_concat.csv", header=T, sep=";", dec = ".")
df<-read.csv("C:/Users/Margherita/Documents/GitHub/SOABB1/Scripts/N72_B.csv", header=T, sep=";", dec = ".")
ph<-read.csv("C:/Users/Margherita/Documents/GitHub/SOABB1/Scripts/N72_phases.csv", header=T, sep=";", dec = ".")
ov <- read.csv("C:/Users/Margherita/Documents/GitHub/SOABB1/Datasets/N72_f0_HNR_RMS.csv", header=T, sep=";", dec = ".")
ov_exp <- ov[ ov$cond != "neutre", ] #remove the neutral group as no modification

### change variable type
dc$BB <- factor(dc$BB)
dc$age_c <- factor(dc$age_c)
dc$age<-dc$age-mean(dc$age) # center age on the mean
df$BB <- factor(df$BB)
df$voc_type <- factor(df$Type_protophone)

### FIGURE 2A STATISTICS : zf0c_Original ~ condition * age
dc$cond <- factor(as.character(dc$cond), levels=c("p100", "neutre","m100"))
contrasts(dc$cond) <- contr.treatment(3)
contrasts(dc$cond)
dc$agex <- dc$age_c
p0 <- lm(zf0c_Original ~ 1, data = dc)
p1 <- lm(zf0c_Original ~ cond, data = dc)
p2 <- lm(zf0c_Original ~ cond + agex, data = dc)
p3 <- lm(zf0c_Original ~ cond * agex, data = dc)
anova(p0,p1,p2,p3)
summary(p1)

## assumptions check
model_parameters(p1)
dev.new(width = 12, height = 9)

performance::check_model(p1)

### FIGURE 2B STATISTICS : zf0c_Original_inv ~ age
t.test(dc$zf0c_Original_inv)

p0 <- lm(zf0c_Original_inv ~ 1, data = dc)
p1 <- lm(zf0c_Original_inv ~ agex, data = dc)
anova(p0,p1)
summary(p0)

dev.new(width = 12, height = 9)

performance::check_model(p0)

### FIGURE 2C & S2 - x="M_minus_NM", ["zf0_Original"] ["f0m_NM"]
# zf0_Original ~ f0m_NM + M_minus_NM
df$f0m_NM<-df$f0m_NM-mean(df$f0m_NM, na.rm = TRUE) # center on the mean
df$M_minus_NM<-df$M_minus_NM-mean(df$M_minus_NM, na.rm = TRUE) # center on the mean

p0 <- lmer(zf0_Original ~ 1 + (1 | BB), data = df)
p1 <- lmer(zf0_Original ~ f0m_NM + (1 | BB), data = df)
p2 <- lmer(zf0_Original ~ f0m_NM + M_minus_NM + (1 | BB), data = df)
anova(p0,p1,p2)
summary(p2)
