rm(list=ls())
library(nlme);
library(lmerTest);
#library(lattice); 
#library(dplyr)
#library(GLMcat)
options(contrasts=c("contr.sum","contr.poly"))
require(nlme)         ## for lme()

### READ DATA
dc<-read.csv("C:/Users/Margherita/Desktop/SOA_Projet/paper1/currbio/aug_2026/aug_submission/final_sub/new_codes/N72_concat.csv", header=T, sep=";", dec = ".")
df<-read.csv("C:/Users/Margherita/Desktop/SOA_Projet/paper1/currbio/aug_2026/aug_submission/final_sub/new_codes/N72_B.csv", header=T, sep=";", dec = ".")
ph<-read.csv("C:/Users/Margherita/Desktop/SOA_Projet/paper1/currbio/aug_2026/aug_submission/final_sub/new_codes/N72_phases.csv", header=T, sep=";", dec = ".")
ov <- read.csv("C:/Users/Margherita/Desktop/for_github_for_paper/SOABB1_working/Datasets/N72_f0_HNR_RMS.csv", header=T, sep=";", dec = ".")
ov_exp <- ov[ ov$cond != "neutre", ] #remove the neutral group as no modification

### change variable type
dc$BB <- factor(dc$BB)
dc$age_c <- factor(dc$age_c)
dc$age<-dc$age-mean(dc$age) # center age on the mean
df$BB <- factor(df$BB)
df$voc_type <- factor(df$Type_protophone)

### CHECK TRANSFORMATION WORKED (SM1)
#set up levels
ph$cond <- factor(as.character(ph$cond), levels=c("p100", "m100","neutre"))
contrasts(ph$cond) <- contr.treatment(3)
contrasts(ph$cond)
ph$phase <- factor(as.character(ph$phase), levels=c("EXP", "BL"))
contrasts(ph$phase) <- contr.treatment(2)
contrasts(ph$phase)
#model comparison
p0 <- lm(M_minus_NM ~ 1, data = ph)
p1 <- lm(M_minus_NM ~ phase, data = ph)
p2 <- lm(M_minus_NM ~ phase + cond, data = ph)
p3 <- lm(M_minus_NM ~ phase * cond, data = ph)
anova(p0,p1,p2,p3)
summary(p3)

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

### FIGURE 2B STATISTICS : zf0c_Original_inv ~ age
t.test(dc$zf0c_Original_inv)

p0 <- lm(zf0c_Original_inv ~ 1, data = dc)
p1 <- lm(zf0c_Original_inv ~ agex, data = dc)
anova(p0,p1)
summary(p0)

### FIGURE 2C & S2 - x="M_minus_NM", ["zf0_Original"] ["f0m_NM"]
# zf0_Original ~ f0m_NM + M_minus_NM
df$f0m_NM<-df$f0m_NM-mean(df$f0m_NM, na.rm = TRUE) # center on the mean
df$M_minus_NM<-df$M_minus_NM-mean(df$M_minus_NM, na.rm = TRUE) # center on the mean

p0 <- lmer(zf0_Original ~ 1 + (1 | BB), data = df)
p1 <- lmer(zf0_Original ~ f0m_NM + (1 | BB), data = df)
p2 <- lmer(zf0_Original ~ f0m_NM + M_minus_NM + (1 | BB), data = df)
anova(p0,p1,p2)
summary(p2)

df$is_margbabbling <- factor(as.character(df$is_margbabbling), levels=c("0", "1"))
contrasts(df$is_margbabbling) <- contr.treatment(2)
p3 <- lmer(zf0_Original ~ f0m_NM + M_minus_NM + is_margbabbling + (1 | BB), data = df)
p4 <- lmer(zf0_Original ~ f0m_NM + M_minus_NM + is_margbabbling + M_minus_NM:is_margbabbling + (1 | BB), data = df)
anova(p0,p1,p2,p3,p4)
summary(p4)

posthoc<-droplevels(df[df$is_margbabbling == 0,])
p0 <- lmer(zf0_Original ~ 1 + (1 | BB), data = posthoc)
p1 <- lmer(zf0_Original ~ f0m_NM + (1 | BB), data = posthoc)
p2 <- lmer(zf0_Original ~ f0m_NM + M_minus_NM + (1 | BB), data = posthoc)
anova(p0,p1,p2)
summary(p2)

## all voc types?
p3 <- lmer(zf0_Original ~ f0m_NM + M_minus_NM + voc_type + (1 | BB), data = df)
p4 <- lmer(zf0_Original ~ f0m_NM + M_minus_NM + voc_type + M_minus_NM:voc_type + (1 | BB), data = df)
anova(p0,p1,p2,p3,p4)
summary(p4)

### S11 : MAGNITUDE MODEL : zf0c_Original_abs ~ condition * age
dc$cond <- factor(as.character(dc$cond), levels=c("p100", "neutre","m100"))
contrasts(dc$cond) <- contr.treatment(3)
contrasts(dc$cond)
dc$agex <- dc$age
p0 <- lm(zf0c_Original_abs ~ 1, data = dc)
p1 <- lm(zf0c_Original_abs ~ cond, data = dc)
p2 <- lm(zf0c_Original_abs ~ cond + agex, data = dc)
p3 <- lm(zf0c_Original_abs ~ cond * agex, data = dc)
anova(p0,p1,p2,p3)
summary(p1)

## Is the vocalisation duration different across conditions? 

#set up contrasts
ov$cond <- factor(as.character(ov$cond), levels=c("neutre", "m100", "p100"))
contrasts(ov$cond) <- contr.treatment(3)
contrasts(ov$cond)

ov$phase <- factor(as.character(ov$phase), levels=c("BL", "EXP"))
contrasts(ov$phase) <- contr.treatment(2)
contrasts(ov$phase)

# upward vs downward

p0 <- lmer(duration ~ 1 + (1 | BB), data = ov_exp, REML = FALSE)
p1 <- lmer(duration ~ phase + (1 | BB), data = ov_exp, REML = FALSE)
p2 <- lmer(duration ~ phase + cond + (1 | BB), data = ov_exp, REML = FALSE)
p3 <- lmer(duration ~ phase + cond + phase*cond + (1 | BB), data = ov_exp, REML = FALSE)
anova(p0,p1,p2,p3) 

# upward vs neutral & downward vs neutral
p0 <- lmer(duration ~ 1 + (1 | BB), data = ov, REML = FALSE)
p1 <- lmer(duration ~ phase + (1 | BB), data = ov, REML = FALSE)
p2 <- lmer(duration ~ phase + cond + (1 | BB), data = ov, REML = FALSE)
p3 <- lmer(duration ~ phase + cond + phase*cond + (1 | BB), data = ov, REML = FALSE)
anova(p0,p1,p2,p3)
