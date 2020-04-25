### This file creates the datafiles necessary to perform machine learning on text data, and it shows how algorithms can be created on local data, based on Speer (2020) ###
### Thus, you would use this if you wanted to create your own Great 8 algorithms based on local data ###
### Once those algorithms are created and saved, you could then use them and score new text using the "1 G8 ORM - use to score text" syntax file

### All these files can be found on the following GitHub: https://github.com/AndrewBoyd10/Great-8-Valence-Dictionary  or OSF (https://osf.io/2wcsn/)
### To begin, copy over the following files from https://github.com/AndrewBoyd10/Great-8-Valence-Dictionary  into a folder on your local computer. This folder will be your working directory:
    #"cal_words" files, which contain word phrases used in random forests models
    # "Great 8 Narrative Dictionary - use": lemmatized theme words used to perform n-gram scoring and to create theme scores
    # "Norm" files: used to adjust scores as last step
    # 27 "RF" objects which contain the algorithms to score data into valence scores
    # Probably easiest to just copy all files into your local working directory

### You will also need to create an R dataframe, saved in your local directory, and named "Text_df" that contains the following variables:
    #Narrative = text (such that "Narrative" is the variable name)
    #Unique_ID_Ratee = ID for a given person rated
    #Ratee_G1_num thru Ratee_G8_num = variables reflecting numerical performance ratings for Great 8 dimensions 1-8.
    #Performance_Composite = variable reflecting performance composite of all numerical performance ratings.
    #Train_filter = variable that randomly segments the file. Segmentation is not incorporated in this syntax and you'll have to do manually if you want to. 
    #in other words, this syntax treats all this as the "training" data

### Finally, you'll need to download the "glove.42B.300d" file from https://nlp.stanford.edu/projects/glove/ and save the file as "glove.42B.300d.txt" in your local folder ###

### Begin by loading relevant packages and setting working directory ###
library(psych)
library(plyr)
library(dplyr)
library(readr)
library(readxl)

library(tm) 
library(stringr)
library(tidytext) 
library(qdap) 
library(quanteda)
library (RWeka)
library(SnowballC)
library(textstem)

library(caret)
library (glmnet)
library(doParallel)
library(parallel)
library(randomForest)

setwd("INSERT_FOLDER_LOCATION") #setting working directory (enter this)#

############################################################################################################################################
############################################################################################################################################
#################################################   Loading Great 8 Theme Dictionaries  ###################################################
############################################################################################################################################
############################################################################################################################################

# Note this is the edited, lemmatized version, which lemmatized the whole dictionary and then removed redundancies
# Will use this in later steps
Dictionary <- readRDS ("Great 8 Narrative Dictionary - use.Rda")
G1_Dic <- dplyr::filter (Dictionary, Dimension=='G1')
G2_Dic <- dplyr::filter (Dictionary, Dimension=='G2')
G3_Dic <- dplyr::filter (Dictionary, Dimension=='G3')
G4_Dic <- dplyr::filter (Dictionary, Dimension=='G4')
G5_Dic <- dplyr::filter (Dictionary, Dimension=='G5')
G6_Dic <- dplyr::filter (Dictionary, Dimension=='G6')
G7_Dic <- dplyr::filter (Dictionary, Dimension=='G7')
G8_Dic <- dplyr::filter (Dictionary, Dimension=='G8')

############################################################################################################################################
############################################################################################################################################
#################################################  Cleaning and Converting Text to Various Files for Later Steps  ###################################################
############################################################################################################################################
############################################################################################################################################

### Loading Text File ###
Text_df <- readRDS ("Text_df.Rda") #loading your initial file. This will be transformed some and then used to create other files
colnames(Text_df)
dim (Text_df)

### Basic Cleaning ###
#general cleaning: some later, some redundant#
Text_df$Narrative <- gsub("\\r\\n"," ", Text_df$Narrative) #r carriage return, n line feed; varies by computer settings#
Text_df$Narrative <- gsub("  *", " ", Text_df$Narrative) #removing extra whitespace
Text_df$Narrative <- gsub("\\<&\\>"," and ", Text_df$Narrative) 
Text_df$Narrative <- gsub("%"," percent ", Text_df$Narrative) 
Text_df$Narrative <- gsub("\\+"," plus ", Text_df$Narrative)
Text_df$Narrative <- gsub("\\$"," dollar ", Text_df$Narrative) 
Text_df$Narrative <- gsub("\\<IQ\\>"," intelligence ", Text_df$Narrative)

##### Saving Text File for Word Embedding Analysis ######
saveRDS(Text_df, "Text_df_embed.Rda") #is applied for word-embeddings later

#more cleaning#
Text_df$Narrative <- gsub("\\!"," exclamation_mark ", Text_df$Narrative)
Text_df$Narrative <- gsub("!"," exclamation_mark ", Text_df$Narrative)
Text_df$Narrative <- gsub("-"," ", Text_df$Narrative) #helps with theme match
Text_df$Narrative <- gsub("\\?"," question_mark ", Text_df$Narrative)

#word and character count#
Text_df$Word_count <- str_count(Text_df$Narrative, "\\w+")
Text_df$Characters <- nchar (Text_df$Narrative)

###
#Sentence Breakdown (note that it automatically lower-cases text) and Additional Cleaning #
###

#splitting to sentences#
sentences <- (dplyr::select (Text_df, Unique_ID_Ratee, Narrative))  %>% 
  unnest_tokens(sentence, Narrative, token = "sentences")
head(sentences) #just taking a quick look#

#count#
sent_sum <- ddply (sentences, "Unique_ID_Ratee", summarize, Ratee_total_sentences = length (sentence))
Text_df <- merge (Text_df, sent_sum, by = "Unique_ID_Ratee", all.x=TRUE)

#Negator: basic function to perform polarity flips#
str_negate <- function(x) {
  x <- gsub("not\\s+", "not_", x)
  x <-   gsub("n't\\s+", " not_", x)
  x <- gsub("cannot\\s+", "not_", x)
  x <- gsub("never\\s+", "never_", x)
  x <- gsub("no\\s+", "no_", x)
  return(x)
}
sentences$sentence <- str_negate (sentences$sentence)
sentences [1:50, c("sentence")] #just taking a quick look#

#removing some gender-specific text#
sentences$sentence <- gsub("\\<he\\>|\\<she\\>|\\<s/he\\>"," PRONOUNOTHER ", sentences$sentence) #left out some intentionally
sentences$sentence <- gsub("\\<i\\>|\\<myself\\>|\\<me\\>"," PRONOUNFIRST ", sentences$sentence) 
sentences$sentence <- gsub("\\<his\\>|\\<hers\\>|\\<her\\>|\\<yours\\>|\\<himself\\>|\\<herself\\>|\\<yourself\\>|\\<thy\\>|\\<thee\\>|\\<his/hers\\>|\\<his/her\\>"," PRONOUNPOSSESSIVEOTHER ", sentences$sentence)  
sentences$sentence <- gsub("\\<mine\\>|\\<my\\>"," PRONOUNPOSSESSIVEFIRST ", sentences$sentence) 
sentences$sentence <- gsub("\\<he's\\>|\\<she's\\>|\\<he'd\\>|\\<she'd\\>|\\<he'll\\>|\\<she'll\\>"," PRONOUNVERBOTHER ", sentences$sentence) 

#given the study task, some employees were described by first letters of an employee's name. For example, "I thought that d did a really great job this year", where "d" is the employee#. 
#I therefore edited these instances too#
sentences$sentence <- gsub("\\<a\\>"," _a_ ", sentences$sentence) #setting apart "a" (i is already handled above) #because people referred to folks as letters
sentences$sentence <- gsub("\\<[A-Za-z]\\>"," PRONOUNOTHER", sentences$sentence) #(a and i already handled above) #because people referred to folks as letters
sentences [1:50, c("sentence")] #just taking a quick look#

#remove stopwords (from 2018)... some of these are already dropped out from prior steps#
custom_stopwords <-  c( "myself",  "i", "in", "on",
                        "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", 
                        "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", 
                        "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", 
                        "having", "do", "does", "did", "doing", "would", "should", "could", "ought", "i'm", "you're", "he's", 
                        "she's", "it's",  "they're", "you've", "we've", "they've", "i'd", "you'd", "he'd", "she'd", 
                        "we'd", "they'd", "i'll", "you'll", "he'll", "she'll", "we'll", "they'll", 
                        "let's", "that's", "who's", "what's", "here's", "there's", "when's", "where's", "why's", "how's", 
                        "_a_", "a", "an", "the", "and", "if", "or", "because", "as", "of", 
                        "at", "by", "for", "with", "about", "into", "through", "during",
                        "to", "from","here", "there", "when", "where", "why", "how", 
                        "both", "each", "such", "only", 
                        "own", "same", "so", "than", "too")
sentences$sentence <- removeWords(sentences$sentence, custom_stopwords)
sentences [1:50, c("sentence")] #just taking a quick look#

#lemmatize#  
sentences$sentence <- textstem::lemmatize_strings(sentences$sentence)
sentences$sentence <- gsub (" _ ", "_", sentences$sentence)

#punctuation#
sentences$sentence <- strip(sentences$sentence, char.keep="_", digit.remove=FALSE)
sentences [1:50, c("sentence")] #just taking a quick look#

#additional cleans#
sentences$sentence <- removeNumbers (sentences$sentence)  
sentences$sentence <- stripWhitespace (sentences$sentence)  
sentences [1:50, c("sentence")] #just taking a quick look#

############################################################################################################################################
############################################################################################################################################
#################################################  Contextualized N-Gram DTMs and Theme Scores  ###################################################
############################################################################################################################################
############################################################################################################################################

#We've already loaded the Great 8 dictionaries, which will now be applied here. Below, searches are made for dictionary-specific phrases and then those sentences are used

###
### G1: Creating DTM File for G1 ###
###
Presence <- vector ()
for (i in G1_Dic$Word) {
  Add <- (grepl (i, sentences$sentence))
  Presence <- cbind (Presence, Add)
}
G1_YesNo <- as.matrix (apply (Presence, 1, max))
G1_Sum <- as.matrix (apply (Presence, 1, sum)) #will apply this later#
sentences_G1 <- cbind (sentences, G1_YesNo)
dim (sentences_G1)

G1_file <- dplyr::select (dplyr::filter (sentences_G1, G1_YesNo==1), Unique_ID_Ratee, sentence)
G1_file <- aggregate (G1_file [,1:2], by=list (G1_file$Unique_ID_Ratee), paste, collapse = " ")
G1_file <- G1_file[,c(1,3)]
names(G1_file) <- c('doc_id', 'text')
G1_file$text <- stripWhitespace (G1_file$text)

#adding in data for those who did not have theme phrases#
x <- dplyr::select(Text_df, Unique_ID_Ratee)
names(x) [1] <- 'doc_id'
G1_file <- merge (x, G1_file, by = "doc_id", all.x=TRUE)
G1_file$text [is.na(G1_file$text)] <- ''
dim (G1_file)

#theme score#
theme <- cbind (dplyr::select (sentences, Unique_ID_Ratee), G1_Sum)
theme <- as.data.frame (ddply (theme, "Unique_ID_Ratee", summarize, G1_Sum = sum(G1_Sum, na.rm=TRUE)))
theme <- dplyr::select (theme, G1_Sum)
head (theme)

Text_df <- cbind (Text_df, theme)
Text_df$G1_Theme_Score <- Text_df$G1_Sum / Text_df$Word_count

#creating dtm#
G1_wordCorpus <- VCorpus(DataframeSource(G1_file))
tokenizer <- function(x) 
  NGramTokenizer(x, Weka_control(min = 1, max = 2)) #bigrams as max
dtm <- DocumentTermMatrix (G1_wordCorpus, control = list (tokenize=tokenizer))
dtm <- removeSparseTerms(dtm, .99) #removing sparse terms that don't occur in at least 1% of the file
dtm <- weightBin(dtm) #dichotomous weighting
dtm_m <- as.matrix(dtm)

Word_count <- as.data.frame(str_count(G1_file$text, "\\w+"))
names (Word_count) [1] <- 'Word_Count_G1'
dtm_m <- cbind (Word_count, dtm_m)
dtm_m <- cbind (Text_df[, c("Unique_ID_Ratee", "Ratee_G1_num", "Train_filter")], dtm_m)
dtm_m [1:10, 1:10]
dim (dtm_m)
saveRDS (dtm_m, "G1_dtm_m.Rda") #saving this dtm for later analysis#

###
### G2: Creating DTM File for G2 (syntax same, just find-replaced w/G2 ###
###
Presence <- vector ()
for (i in G2_Dic$Word) {
  Add <- (grepl (i, sentences$sentence))
  Presence <- cbind (Presence, Add)
}
G2_YesNo <- as.matrix (apply (Presence, 1, max))
G2_Sum <- as.matrix (apply (Presence, 1, sum)) #will apply this later#
sentences_G2 <- cbind (sentences, G2_YesNo)
dim (sentences_G2)

G2_file <- dplyr::select (dplyr::filter (sentences_G2, G2_YesNo==1), Unique_ID_Ratee, sentence)
G2_file <- aggregate (G2_file [,1:2], by=list (G2_file$Unique_ID_Ratee), paste, collapse = " ")
G2_file <- G2_file[,c(1,3)]
names(G2_file) <- c('doc_id', 'text')
G2_file$text <- stripWhitespace (G2_file$text)

#adding in data for those who did not have theme phrases#
x <- dplyr::select(Text_df, Unique_ID_Ratee)
names(x) [1] <- 'doc_id'
G2_file <- merge (x, G2_file, by = "doc_id", all.x=TRUE)
G2_file$text [is.na(G2_file$text)] <- ''
dim (G2_file)

#theme score#
theme <- cbind (dplyr::select (sentences, Unique_ID_Ratee), G2_Sum)
theme <- as.data.frame (ddply (theme, "Unique_ID_Ratee", summarize, G2_Sum = sum(G2_Sum, na.rm=TRUE)))
theme <- dplyr::select (theme, G2_Sum)
head (theme)

Text_df <- cbind (Text_df, theme)
Text_df$G2_Theme_Score <- Text_df$G2_Sum / Text_df$Word_count

#creating dtm#
G2_wordCorpus <- VCorpus(DataframeSource(G2_file))
tokenizer <- function(x) 
  NGramTokenizer(x, Weka_control(min = 1, max = 2)) #bigrams as max
dtm <- DocumentTermMatrix (G2_wordCorpus, control = list (tokenize=tokenizer))
dtm <- removeSparseTerms(dtm, .99) #removing sparse terms that don't occur in at least 1% of the file
dtm <- weightBin(dtm) #dichotomous weighting
dtm_m <- as.matrix(dtm)

Word_count <- as.data.frame(str_count(G2_file$text, "\\w+"))
names (Word_count) [1] <- 'Word_Count_G2'
dtm_m <- cbind (Word_count, dtm_m)
dtm_m <- cbind (Text_df[, c("Unique_ID_Ratee", "Ratee_G2_num", "Train_filter")], dtm_m)
dtm_m [1:10, 1:10]
dim (dtm_m)
saveRDS (dtm_m, "G2_dtm_m.Rda") #saving this dtm for later analysis#

###
### G3: Creating DTM File for G3 (syntax same, just find-replaced w/G3 ###
###
Presence <- vector ()
for (i in G3_Dic$Word) {
  Add <- (grepl (i, sentences$sentence))
  Presence <- cbind (Presence, Add)
}
G3_YesNo <- as.matrix (apply (Presence, 1, max))
G3_Sum <- as.matrix (apply (Presence, 1, sum)) #will apply this later#
sentences_G3 <- cbind (sentences, G3_YesNo)
dim (sentences_G3)

G3_file <- dplyr::select (dplyr::filter (sentences_G3, G3_YesNo==1), Unique_ID_Ratee, sentence)
G3_file <- aggregate (G3_file [,1:2], by=list (G3_file$Unique_ID_Ratee), paste, collapse = " ")
G3_file <- G3_file[,c(1,3)]
names(G3_file) <- c('doc_id', 'text')
G3_file$text <- stripWhitespace (G3_file$text)

#adding in data for those who did not have theme phrases#
x <- dplyr::select(Text_df, Unique_ID_Ratee)
names(x) [1] <- 'doc_id'
G3_file <- merge (x, G3_file, by = "doc_id", all.x=TRUE)
G3_file$text [is.na(G3_file$text)] <- ''
dim (G3_file)

#theme score#
theme <- cbind (dplyr::select (sentences, Unique_ID_Ratee), G3_Sum)
theme <- as.data.frame (ddply (theme, "Unique_ID_Ratee", summarize, G3_Sum = sum(G3_Sum, na.rm=TRUE)))
theme <- dplyr::select (theme, G3_Sum)
head (theme)

Text_df <- cbind (Text_df, theme)
Text_df$G3_Theme_Score <- Text_df$G3_Sum / Text_df$Word_count

#creating dtm#
G3_wordCorpus <- VCorpus(DataframeSource(G3_file))
tokenizer <- function(x) 
  NGramTokenizer(x, Weka_control(min = 1, max = 2)) #bigrams as max
dtm <- DocumentTermMatrix (G3_wordCorpus, control = list (tokenize=tokenizer))
dtm <- removeSparseTerms(dtm, .99) #removing sparse terms that don't occur in at least 1% of the file
dtm <- weightBin(dtm) #dichotomous weighting
dtm_m <- as.matrix(dtm)

Word_count <- as.data.frame(str_count(G3_file$text, "\\w+"))
names (Word_count) [1] <- 'Word_Count_G3'
dtm_m <- cbind (Word_count, dtm_m)
dtm_m <- cbind (Text_df[, c("Unique_ID_Ratee", "Ratee_G3_num", "Train_filter")], dtm_m)
dtm_m [1:10, 1:10]
dim (dtm_m)
saveRDS (dtm_m, "G3_dtm_m.Rda") #saving this dtm for later analysis#

###
### G4: Creating DTM File for G4 (syntax same, just find-replaced w/G4 ###
###
Presence <- vector ()
for (i in G4_Dic$Word) {
  Add <- (grepl (i, sentences$sentence))
  Presence <- cbind (Presence, Add)
}
G4_YesNo <- as.matrix (apply (Presence, 1, max))
G4_Sum <- as.matrix (apply (Presence, 1, sum)) #will apply this later#
sentences_G4 <- cbind (sentences, G4_YesNo)
dim (sentences_G4)

G4_file <- dplyr::select (dplyr::filter (sentences_G4, G4_YesNo==1), Unique_ID_Ratee, sentence)
G4_file <- aggregate (G4_file [,1:2], by=list (G4_file$Unique_ID_Ratee), paste, collapse = " ")
G4_file <- G4_file[,c(1,3)]
names(G4_file) <- c('doc_id', 'text')
G4_file$text <- stripWhitespace (G4_file$text)

#adding in data for those who did not have theme phrases#
x <- dplyr::select(Text_df, Unique_ID_Ratee)
names(x) [1] <- 'doc_id'
G4_file <- merge (x, G4_file, by = "doc_id", all.x=TRUE)
G4_file$text [is.na(G4_file$text)] <- ''
dim (G4_file)

#theme score#
theme <- cbind (dplyr::select (sentences, Unique_ID_Ratee), G4_Sum)
theme <- as.data.frame (ddply (theme, "Unique_ID_Ratee", summarize, G4_Sum = sum(G4_Sum, na.rm=TRUE)))
theme <- dplyr::select (theme, G4_Sum)
head (theme)

Text_df <- cbind (Text_df, theme)
Text_df$G4_Theme_Score <- Text_df$G4_Sum / Text_df$Word_count

#creating dtm#
G4_wordCorpus <- VCorpus(DataframeSource(G4_file))
tokenizer <- function(x) 
  NGramTokenizer(x, Weka_control(min = 1, max = 2)) #bigrams as max
dtm <- DocumentTermMatrix (G4_wordCorpus, control = list (tokenize=tokenizer))
dtm <- removeSparseTerms(dtm, .99) #removing sparse terms that don't occur in at least 1% of the file
dtm <- weightBin(dtm) #dichotomous weighting
dtm_m <- as.matrix(dtm)

Word_count <- as.data.frame(str_count(G4_file$text, "\\w+"))
names (Word_count) [1] <- 'Word_Count_G4'
dtm_m <- cbind (Word_count, dtm_m)
dtm_m <- cbind (Text_df[, c("Unique_ID_Ratee", "Ratee_G4_num", "Train_filter")], dtm_m)
dtm_m [1:10, 1:10]
dim (dtm_m)
saveRDS (dtm_m, "G4_dtm_m.Rda") #saving this dtm for later analysis#

###
### G5: Creating DTM File for G5 (syntax same, just find-replaced w/G5 ###
###
Presence <- vector ()
for (i in G5_Dic$Word) {
  Add <- (grepl (i, sentences$sentence))
  Presence <- cbind (Presence, Add)
}
G5_YesNo <- as.matrix (apply (Presence, 1, max))
G5_Sum <- as.matrix (apply (Presence, 1, sum)) #will apply this later#
sentences_G5 <- cbind (sentences, G5_YesNo)
dim (sentences_G5)

G5_file <- dplyr::select (dplyr::filter (sentences_G5, G5_YesNo==1), Unique_ID_Ratee, sentence)
G5_file <- aggregate (G5_file [,1:2], by=list (G5_file$Unique_ID_Ratee), paste, collapse = " ")
G5_file <- G5_file[,c(1,3)]
names(G5_file) <- c('doc_id', 'text')
G5_file$text <- stripWhitespace (G5_file$text)

#adding in data for those who did not have theme phrases#
x <- dplyr::select(Text_df, Unique_ID_Ratee)
names(x) [1] <- 'doc_id'
G5_file <- merge (x, G5_file, by = "doc_id", all.x=TRUE)
G5_file$text [is.na(G5_file$text)] <- ''
dim (G5_file)

#theme score#
theme <- cbind (dplyr::select (sentences, Unique_ID_Ratee), G5_Sum)
theme <- as.data.frame (ddply (theme, "Unique_ID_Ratee", summarize, G5_Sum = sum(G5_Sum, na.rm=TRUE)))
theme <- dplyr::select (theme, G5_Sum)
head (theme)

Text_df <- cbind (Text_df, theme)
Text_df$G5_Theme_Score <- Text_df$G5_Sum / Text_df$Word_count

#creating dtm#
G5_wordCorpus <- VCorpus(DataframeSource(G5_file))
tokenizer <- function(x) 
  NGramTokenizer(x, Weka_control(min = 1, max = 2)) #bigrams as max
dtm <- DocumentTermMatrix (G5_wordCorpus, control = list (tokenize=tokenizer))
dtm <- removeSparseTerms(dtm, .99) #removing sparse terms that don't occur in at least 1% of the file
dtm <- weightBin(dtm) #dichotomous weighting
dtm_m <- as.matrix(dtm)

Word_count <- as.data.frame(str_count(G5_file$text, "\\w+"))
names (Word_count) [1] <- 'Word_Count_G5'
dtm_m <- cbind (Word_count, dtm_m)
dtm_m <- cbind (Text_df[, c("Unique_ID_Ratee", "Ratee_G5_num", "Train_filter")], dtm_m)
dtm_m [1:10, 1:10]
dim (dtm_m)
saveRDS (dtm_m, "G5_dtm_m.Rda") #saving this dtm for later analysis#

###
### G6: Creating DTM File for G6 (syntax same, just find-replaced w/G6 ###
###
Presence <- vector ()
for (i in G6_Dic$Word) {
  Add <- (grepl (i, sentences$sentence))
  Presence <- cbind (Presence, Add)
}
G6_YesNo <- as.matrix (apply (Presence, 1, max))
G6_Sum <- as.matrix (apply (Presence, 1, sum)) #will apply this later#
sentences_G6 <- cbind (sentences, G6_YesNo)
dim (sentences_G6)

G6_file <- dplyr::select (dplyr::filter (sentences_G6, G6_YesNo==1), Unique_ID_Ratee, sentence)
G6_file <- aggregate (G6_file [,1:2], by=list (G6_file$Unique_ID_Ratee), paste, collapse = " ")
G6_file <- G6_file[,c(1,3)]
names(G6_file) <- c('doc_id', 'text')
G6_file$text <- stripWhitespace (G6_file$text)

#adding in data for those who did not have theme phrases#
x <- dplyr::select(Text_df, Unique_ID_Ratee)
names(x) [1] <- 'doc_id'
G6_file <- merge (x, G6_file, by = "doc_id", all.x=TRUE)
G6_file$text [is.na(G6_file$text)] <- ''
dim (G6_file)

#theme score#
theme <- cbind (dplyr::select (sentences, Unique_ID_Ratee), G6_Sum)
theme <- as.data.frame (ddply (theme, "Unique_ID_Ratee", summarize, G6_Sum = sum(G6_Sum, na.rm=TRUE)))
theme <- dplyr::select (theme, G6_Sum)
head (theme)

Text_df <- cbind (Text_df, theme)
Text_df$G6_Theme_Score <- Text_df$G6_Sum / Text_df$Word_count

#creating dtm#
G6_wordCorpus <- VCorpus(DataframeSource(G6_file))
tokenizer <- function(x) 
  NGramTokenizer(x, Weka_control(min = 1, max = 2)) #bigrams as max
dtm <- DocumentTermMatrix (G6_wordCorpus, control = list (tokenize=tokenizer))
dtm <- removeSparseTerms(dtm, .99) #removing sparse terms that don't occur in at least 1% of the file
dtm <- weightBin(dtm) #dichotomous weighting
dtm_m <- as.matrix(dtm)

Word_count <- as.data.frame(str_count(G6_file$text, "\\w+"))
names (Word_count) [1] <- 'Word_Count_G6'
dtm_m <- cbind (Word_count, dtm_m)
dtm_m <- cbind (Text_df[, c("Unique_ID_Ratee", "Ratee_G6_num", "Train_filter")], dtm_m)
dtm_m [1:10, 1:10]
dim (dtm_m)
saveRDS (dtm_m, "G6_dtm_m.Rda") #saving this dtm for later analysis#

###
### G7: Creating DTM File for G7 (syntax same, just find-replaced w/G7 ###
###
Presence <- vector ()
for (i in G7_Dic$Word) {
  Add <- (grepl (i, sentences$sentence))
  Presence <- cbind (Presence, Add)
}
G7_YesNo <- as.matrix (apply (Presence, 1, max))
G7_Sum <- as.matrix (apply (Presence, 1, sum)) #will apply this later#
sentences_G7 <- cbind (sentences, G7_YesNo)
dim (sentences_G7)

G7_file <- dplyr::select (dplyr::filter (sentences_G7, G7_YesNo==1), Unique_ID_Ratee, sentence)
G7_file <- aggregate (G7_file [,1:2], by=list (G7_file$Unique_ID_Ratee), paste, collapse = " ")
G7_file <- G7_file[,c(1,3)]
names(G7_file) <- c('doc_id', 'text')
G7_file$text <- stripWhitespace (G7_file$text)

#adding in data for those who did not have theme phrases#
x <- dplyr::select(Text_df, Unique_ID_Ratee)
names(x) [1] <- 'doc_id'
G7_file <- merge (x, G7_file, by = "doc_id", all.x=TRUE)
G7_file$text [is.na(G7_file$text)] <- ''
dim (G7_file)

#theme score#
theme <- cbind (dplyr::select (sentences, Unique_ID_Ratee), G7_Sum)
theme <- as.data.frame (ddply (theme, "Unique_ID_Ratee", summarize, G7_Sum = sum(G7_Sum, na.rm=TRUE)))
theme <- dplyr::select (theme, G7_Sum)
head (theme)

Text_df <- cbind (Text_df, theme)
Text_df$G7_Theme_Score <- Text_df$G7_Sum / Text_df$Word_count

#creating dtm#
G7_wordCorpus <- VCorpus(DataframeSource(G7_file))
tokenizer <- function(x) 
  NGramTokenizer(x, Weka_control(min = 1, max = 2)) #bigrams as max
dtm <- DocumentTermMatrix (G7_wordCorpus, control = list (tokenize=tokenizer))
dtm <- removeSparseTerms(dtm, .99) #removing sparse terms that don't occur in at least 1% of the file
dtm <- weightBin(dtm) #dichotomous weighting
dtm_m <- as.matrix(dtm)

Word_count <- as.data.frame(str_count(G7_file$text, "\\w+"))
names (Word_count) [1] <- 'Word_Count_G7'
dtm_m <- cbind (Word_count, dtm_m)
dtm_m <- cbind (Text_df[, c("Unique_ID_Ratee", "Ratee_G7_num", "Train_filter")], dtm_m)
dtm_m [1:10, 1:10]
dim (dtm_m)
saveRDS (dtm_m, "G7_dtm_m.Rda") #saving this dtm for later analysis#

###
### G8: Creating DTM File for G8 (syntax same, just find-replaced w/G8 ###
###
Presence <- vector ()
for (i in G8_Dic$Word) {
  Add <- (grepl (i, sentences$sentence))
  Presence <- cbind (Presence, Add)
}
G8_YesNo <- as.matrix (apply (Presence, 1, max))
G8_Sum <- as.matrix (apply (Presence, 1, sum)) #will apply this later#
sentences_G8 <- cbind (sentences, G8_YesNo)
dim (sentences_G8)

G8_file <- dplyr::select (dplyr::filter (sentences_G8, G8_YesNo==1), Unique_ID_Ratee, sentence)
G8_file <- aggregate (G8_file [,1:2], by=list (G8_file$Unique_ID_Ratee), paste, collapse = " ")
G8_file <- G8_file[,c(1,3)]
names(G8_file) <- c('doc_id', 'text')
G8_file$text <- stripWhitespace (G8_file$text)

#adding in data for those who did not have theme phrases#
x <- dplyr::select(Text_df, Unique_ID_Ratee)
names(x) [1] <- 'doc_id'
G8_file <- merge (x, G8_file, by = "doc_id", all.x=TRUE)
G8_file$text [is.na(G8_file$text)] <- ''
dim (G8_file)

#theme score#
theme <- cbind (dplyr::select (sentences, Unique_ID_Ratee), G8_Sum)
theme <- as.data.frame (ddply (theme, "Unique_ID_Ratee", summarize, G8_Sum = sum(G8_Sum, na.rm=TRUE)))
theme <- dplyr::select (theme, G8_Sum)
head (theme)

Text_df <- cbind (Text_df, theme)
Text_df$G8_Theme_Score <- Text_df$G8_Sum / Text_df$Word_count

#creating dtm#
G8_wordCorpus <- VCorpus(DataframeSource(G8_file))
tokenizer <- function(x) 
  NGramTokenizer(x, Weka_control(min = 1, max = 2)) #bigrams as max
dtm <- DocumentTermMatrix (G8_wordCorpus, control = list (tokenize=tokenizer))
dtm <- removeSparseTerms(dtm, .99) #removing sparse terms that don't occur in at least 1% of the file
dtm <- weightBin(dtm) #dichotomous weighting
dtm_m <- as.matrix(dtm)

Word_count <- as.data.frame(str_count(G8_file$text, "\\w+"))
names (Word_count) [1] <- 'Word_Count_G8'
dtm_m <- cbind (Word_count, dtm_m)
dtm_m <- cbind (Text_df[, c("Unique_ID_Ratee", "Ratee_G8_num", "Train_filter")], dtm_m)
dtm_m [1:10, 1:10]
dim (dtm_m)
saveRDS (dtm_m, "G8_dtm_m.Rda") #saving this dtm for later analysis#

###
### Overall: Creating DTM File for performance composite ###
###
Overall_file <- sentences
Overall_file <- aggregate (Overall_file [,2], by=list (Overall_file$Unique_ID_Ratee), paste, collapse = " ")
names(Overall_file) <- c('doc_id', 'text')
Overall_file$text <- stripWhitespace (Overall_file$text)
dim (Overall_file)

#creating dtm#
Overall_wordCorpus <- VCorpus(DataframeSource(Overall_file))
tokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 1, max = 2))
dtm <- DocumentTermMatrix (Overall_wordCorpus, control = list (tokenize=tokenizer))
dtm <- removeSparseTerms(dtm, .99)
dtm <- weightBin(dtm) #dichotomous weighting
dtm_m <- as.matrix(dtm)

Word_count <- str_count(Overall_file$text, "\\w+")
Word_count <- as.data.frame(Word_count)
names (Word_count) [1] <- 'Word_Count_Overall'
dtm_m <- cbind (Word_count, dtm_m)
dtm_m <- cbind (Text_df[, c("Unique_ID_Ratee", "Performance_Composite", "Train_filter")], dtm_m)
dtm_m [1:10, 1:10]
dim (dtm_m)
saveRDS (dtm_m, "Overall_dtm_m.Rda")

###
#Saving the base Text File#
###
colnames(Text_df)
dim (Text_df)
saveRDS(Text_df, "Text_df.Rda") 

############################################################################################################################################
############################################################################################################################################
#################################################   Glove Setup  ###################################################
############################################################################################################################################
############################################################################################################################################

Text_df <- readRDS ("Text_df_embed.Rda")
colnames(Text_df)

Text_df$Word_count <- str_count(Text_df$Narrative, "\\w+")
Text_df$Characters <- nchar (Text_df$Narrative)

###
### Working on local data ###
###
count <- Text_df$Word_count
file <- dplyr::select(Text_df, Unique_ID_Ratee, Narrative)
names(file) <- c('doc_id', 'text')
file$text <- tolower(file$text)
file$text <- replace_contraction (file$text) #some additional minor cleaning
file$text <- strip (file$text, char.keep= c("_", "!", "?"), digit.remove=FALSE) #some additional minor cleaning 
file$text <- removeNumbers (file$text)  #some additional minor cleaning 
file$text <- stripWhitespace (file$text)  #some additional minor cleaning 

### Create DTM ### 
Corpus <- VCorpus(DataframeSource(file))
tokenizer <- function(x) 
  NGramTokenizer(x, Weka_control(min = 1, max = 1))
dtm <- DocumentTermMatrix (Corpus, control = list (tokenize=tokenizer))
dtm <- removeSparseTerms(dtm, 0.99) #requiring they are in at least 1%
dtm_ordered <- dtm  [order(rownames(dtm)),]    #ordering variables
ID_var <- as.data.frame (rownames(dtm)) #creating ID variable for matching later
names(ID_var) [1] <- 'Unique_ID_Ratee' #creating ID variable for matching later
dtm <- as.matrix (apply (dtm, 2, scale)) #normalizing
dtm [1:10, 1:10]
dim (dtm)

### storing possible words so that I can filter down the GloVe embedding file ###
names <- as.data.frame (colnames(dtm))
colnames(names)[1] <- 'names'

###
### Pulling in Glove Data ###
###
#To begin, download one of the GloVe files with 300 dimensions from https://nlp.stanford.edu/projects/glove/, run by Pennington, Socher, & Manning. I'm using the "glove.42B.300d" file
#took file and processed using the following: https://gist.github.com/tjvananne/8b0e7df7dcad414e8e6d5bf3947439a9. this will take some time

#input .txt file, exports list of list of values and character vector of names (words)
proc_pretrained_vec <- function(p_vec) {
  # initialize space for values and the names of each word in vocab
  vals <- vector(mode = "list", length(p_vec))
  names <- character(length(p_vec))
  # loop through to gather values and names of each word
  for(i in 1:length(p_vec)) {
    if(i %% 1000 == 0) {print(i)}
    this_vec <- p_vec[i]
    this_vec_unlisted <- unlist(strsplit(this_vec, " "))
    this_vec_values <- as.numeric(this_vec_unlisted[-1])
    this_vec_name <- this_vec_unlisted[1]
    vals[[i]] <- this_vec_values
    names[[i]] <- this_vec_name
  }
  # convert lists to data.frame and attach the names
  glove <- data.frame(vals)
  names(glove) <- names
  return(glove)
}
g6b_300 <- scan(file = "F:/Projects 2/Job Performance and Leadership/6. Text Analysis 2/1. Rater Effects/Working Analyses - Dict/glove.6B/glove.42B.300d.txt", what="", sep="\n")
# call the function to convert the raw GloVe vector to data.frame
glove.300 <- proc_pretrained_vec(g6b_300)  # this is the actual function call
dim (glove.300)

#####
#### Filtering that down and scoring the vectors on local data ###
#####
glove.300 <- glove.300 [,colnames(glove.300) %in% names$names]
glove.300 <- t(glove.300)
colnames(glove.300) <- paste ("Glove",c(seq(1:ncol(glove.300))),sep="_")
glove.300 [1:10, 1:10]
dim (glove.300)

dtm <- dtm [, colnames(dtm) %in% rownames(glove.300)] #should be unecessary, but in case a word wasn't matched to GloVe database (in which case you might want to check formatting)
dim (dtm)

glove.300 <- glove.300 [order(rownames(glove.300)),] #ordering variables
head(rownames(glove.300))
glove.300 <- as.matrix (apply (glove.300, 2, scale)) #normalizing
glove.300 [1:10, 1:10]
dim (glove.300)

#####
#### Scoring and Finishing Setup ###
#####
embeddings <- as.data.frame (as.matrix (dtm %*% glove.300)) #uses product of dtm and embedding matrix, average of vectors of words#
dim(embeddings) #N by k (300)
embeddings <- embeddings / count #controlling for number of words#
embeddings <- cbind (ID_var, embeddings)
embeddings [1:10, 1:10]

#merging base file with embedding variables#
Text_df <- merge (Text_df, embeddings, by = "Unique_ID_Ratee", all.x=TRUE) 
colnames(Text_df)
dim(Text_df)
saveRDS (dplyr::select(Text_df, Unique_ID_Ratee, Ratee_G1_num:Ratee_G8_num, Performance_Composite, Train_filter, Word_count, Glove_1:Glove_300), "Embed_df.Rda") #saving embedding file

############################################################################################################################################
############################################################################################################################################
#################################################   Building Contextualized N-gram Models  ###################################################
############################################################################################################################################
############################################################################################################################################

#### Loading Primary File from Earlier ####
Text_df <- readRDS ("Text_df.Rda")
colnames(Text_df)
dim (Text_df)

######
###### G1 #####
#######
df <- readRDS ("G1_dtm_m.Rda")
dim (df)
outcome_df <- df [,1:3] #assumes first variable is ID, second variable is performance rating, and third variable is the sample indicator
df <- as.matrix (df [,4:ncol(df)]) #selecting the predictor variables
df <- as.data.frame (apply (df, 2, scale)) #normalizing
df [1:10, 1:10]
outcome_df [1:10,]

#### correlational filter (bootstrapping to reduce sample dependency) ####
set.seed(11111)
dat <- cbind(as.matrix(outcome_df[,2]), df)
N <- 1000
R <- 50 # running 50 bootstrapped samples and taking the average
cor_df  <-   data.frame(matrix(NA, nrow = ncol(df), ncol = R))
for (i in 1:R) {
  x <- dat [sample(nrow(dat),size=nrow(dat),replace=TRUE),]
  res <- corr.test(x[,2:ncol(x)], as.matrix(x[,1]))$r
  res [,1] [is.na(res[,1])] <- 0
  cor_df [, i] <- res [,1]
}
cor_df$r <- rowMeans(cor_df[,1:R])
cor_df <- dplyr::select (cor_df, r)
cor_df$abs_r <- abs(cor_df[,1])
cor_df <- cbind (cor_df, colnames(df))
cor_df <- dplyr::arrange(cor_df, desc(abs_r))
cor_df <- dplyr::filter(cor_df, abs_r >= .05) 
cor_df 
cal_words_ngrams_G1_CAL <- as.data.frame(cor_df[,3])

#### Random Forest  ####
df <- df [,colnames(df) %in% cal_words_ngrams_G1_CAL[,1]]
modeling_df <- cbind (outcome_df[,2], df)
colnames(modeling_df) <- paste(colnames(modeling_df), "_rf", sep = "")
colnames(modeling_df) <- gsub(" ", "__", colnames(modeling_df))
colnames(modeling_df)[1] <- 'Ratee_G1_num'
modeling_df <- as.data.frame (apply (modeling_df, 2, scale)) #normalizing entire dataset
modeling_df [1:5, 1:10]
dim (modeling_df)

num_cores <- detectCores() - 1 # Run in parallel across processors on local machine
cl.spec <- rep("localhost", num_cores) # Run in parallel across processors on local machine
cl <- parallel::makeCluster(cl.spec) # Run in parallel across processors on local machine
registerDoParallel(cl, cores=3)
set.seed(333333)
#run below to determine the optimal mtry, or number of variables per node
#Note: explored optimal number of trees in separate analyses, with anywhere from 2-3k being about the same performance
#fit  <- tuneRF(modeling_df[,2:ncol(modeling_df)], y=modeling_df[,1], mtryStart=(ncol(modeling_df)/5),ntreeTry=2000,stepFactor=1.25,improve=.00001) 
#fit
rf_model <-  randomForest (Ratee_G1_num ~ ., data = modeling_df, nodesize=5, ntree=2000, mtry= 11) #manually input the optimal number of variables from step above
rf_model
saveRDS(rf_model, "RF_fit_ngram_G1_overallCAL.Rda")
stopCluster(cl)
closeAllConnections()
registerDoSEQ()

######
###### G2 #####
#######
df <- readRDS ("G2_dtm_m.Rda")
outcome_df <- df [,1:3] #assumes first variable is ID, second variable is performance rating, and third variable is the sample indicator
df <- as.matrix (df [,4:ncol(df)]) #selecting the predictor variables
df <- as.data.frame (apply (df, 2, scale)) #normalizing
df [1:10, 1:10]
outcome_df [1:10,]

#### correlational filter (bootstrapping to reduce sample dependency) ####
set.seed(11111)
dat <- cbind(as.matrix(outcome_df[,2]), df)
N <- 1000
R <- 50 # running 50 bootstrapped samples and taking the average
cor_df  <-   data.frame(matrix(NA, nrow = ncol(df), ncol = R))
for (i in 1:R) {
  x <- dat [sample(nrow(dat),size=nrow(dat),replace=TRUE),]
  res <- corr.test(x[,2:ncol(x)], as.matrix(x[,1]))$r
  res [,1] [is.na(res[,1])] <- 0
  cor_df [, i] <- res [,1]
}
cor_df$r <- rowMeans(cor_df[,1:R])
cor_df <- dplyr::select (cor_df, r)
cor_df$abs_r <- abs(cor_df[,1])
cor_df <- cbind (cor_df, colnames(df))
cor_df <- dplyr::arrange(cor_df, desc(abs_r))
cor_df <- dplyr::filter(cor_df, abs_r >= .05) 
cor_df 
cal_words_ngrams_G2_CAL <- as.data.frame(cor_df[,3])

#### Random Forest  ####
df <- df [,colnames(df) %in% cal_words_ngrams_G2_CAL[,1]]
modeling_df <- cbind (outcome_df[,2], df)
colnames(modeling_df) <- paste(colnames(modeling_df), "_rf", sep = "")
colnames(modeling_df) <- gsub(" ", "__", colnames(modeling_df))
colnames(modeling_df)[1] <- 'Ratee_G2_num'
modeling_df <- as.data.frame (apply (modeling_df, 2, scale)) #normalizing entire dataset
modeling_df [1:5, 1:10]
dim (modeling_df)

num_cores <- detectCores() - 1 # Run in parallel across processors on local machine
cl.spec <- rep("localhost", num_cores) # Run in parallel across processors on local machine
cl <- parallel::makeCluster(cl.spec) # Run in parallel across processors on local machine
registerDoParallel(cl, cores=3)
set.seed(333333)
#run below to determine the optimal mtry, or number of variables per node
#Note: explored optimal number of trees in separate analyses, with anywhere from 2-3k being about the same performance
#fit  <- tuneRF(modeling_df[,2:ncol(modeling_df)], y=modeling_df[,1], mtryStart=(ncol(modeling_df)/5),ntreeTry=2000,stepFactor=1.25,improve=.00001) 
#fit
rf_model <-  randomForest (Ratee_G2_num ~ ., data = modeling_df, nodesize=5, ntree=2000, mtry= 15) #manually input the optimal number of variables from step above
rf_model
saveRDS(rf_model, "RF_fit_ngram_G2_overallCAL.Rda")
stopCluster(cl)
closeAllConnections()
registerDoSEQ()

######
###### G3 #####
#######
df <- readRDS ("G3_dtm_m.Rda")
outcome_df <- df [,1:3] #assumes first variable is ID, second variable is performance rating, and third variable is the sample indicator
df <- as.matrix (df [,4:ncol(df)]) #selecting the predictor variables
df <- as.data.frame (apply (df, 2, scale)) #normalizing
df [1:10, 1:10]
outcome_df [1:10,]

#### correlational filter (bootstrapping to reduce sample dependency) ####
set.seed(11111)
dat <- cbind(as.matrix(outcome_df[,2]), df)
N <- 1000
R <- 50 # running 50 bootstrapped samples and taking the average
cor_df  <-   data.frame(matrix(NA, nrow = ncol(df), ncol = R))
for (i in 1:R) {
  x <- dat [sample(nrow(dat),size=nrow(dat),replace=TRUE),]
  res <- corr.test(x[,2:ncol(x)], as.matrix(x[,1]))$r
  res [,1] [is.na(res[,1])] <- 0
  cor_df [, i] <- res [,1]
}
cor_df$r <- rowMeans(cor_df[,1:R])
cor_df <- dplyr::select (cor_df, r)
cor_df$abs_r <- abs(cor_df[,1])
cor_df <- cbind (cor_df, colnames(df))
cor_df <- dplyr::arrange(cor_df, desc(abs_r))
cor_df <- dplyr::filter(cor_df, abs_r >= .05) 
cor_df 
cal_words_ngrams_G3_CAL <- as.data.frame(cor_df[,3])

#### Random Forest  ####
df <- df [,colnames(df) %in% cal_words_ngrams_G3_CAL[,1]]
modeling_df <- cbind (outcome_df[,2], df)
colnames(modeling_df) <- paste(colnames(modeling_df), "_rf", sep = "")
colnames(modeling_df) <- gsub(" ", "__", colnames(modeling_df))
colnames(modeling_df)[1] <- 'Ratee_G3_num'
modeling_df <- as.data.frame (apply (modeling_df, 2, scale)) #normalizing entire dataset
modeling_df [1:5, 1:10]
dim (modeling_df)

num_cores <- detectCores() - 1 # Run in parallel across processors on local machine
cl.spec <- rep("localhost", num_cores) # Run in parallel across processors on local machine
cl <- parallel::makeCluster(cl.spec) # Run in parallel across processors on local machine
registerDoParallel(cl, cores=3)
set.seed(333333)
#run below to determine the optimal mtry, or number of variables per node
#Note: explored optimal number of trees in separate analyses, with anywhere from 2-3k being about the same performance
#fit  <- tuneRF(modeling_df[,2:ncol(modeling_df)], y=modeling_df[,1], mtryStart=(ncol(modeling_df)/5),ntreeTry=2000,stepFactor=1.25,improve=.00001) 
#fit
rf_model <-  randomForest (Ratee_G3_num ~ ., data = modeling_df, nodesize=5, ntree=2000, mtry= 16) #manually input the optimal number of variables from step above
rf_model
saveRDS(rf_model, "RF_fit_ngram_G3_overallCAL.Rda")
stopCluster(cl)
closeAllConnections()
registerDoSEQ()

######
###### G4 #####
#######
df <- readRDS ("G4_dtm_m.Rda")
dim (df)
outcome_df <- df [,1:3] #assumes first variable is ID, second variable is performance rating, and third variable is the sample indicator
df <- as.matrix (df [,4:ncol(df)]) #selecting the predictor variables
df <- as.data.frame (apply (df, 2, scale)) #normalizing
df [1:10, 1:10]
outcome_df [1:10,]

#### correlational filter (bootstrapping to reduce sample dependency) ####
set.seed(11111)
dat <- cbind(as.matrix(outcome_df[,2]), df)
N <- 1000
R <- 50 # running 50 bootstrapped samples and taking the average
cor_df  <-   data.frame(matrix(NA, nrow = ncol(df), ncol = R))
for (i in 1:R) {
  x <- dat [sample(nrow(dat),size=nrow(dat),replace=TRUE),]
  res <- corr.test(x[,2:ncol(x)], as.matrix(x[,1]))$r
  res [,1] [is.na(res[,1])] <- 0
  cor_df [, i] <- res [,1]
}
cor_df$r <- rowMeans(cor_df[,1:R])
cor_df <- dplyr::select (cor_df, r)
cor_df$abs_r <- abs(cor_df[,1])
cor_df <- cbind (cor_df, colnames(df))
cor_df <- dplyr::arrange(cor_df, desc(abs_r))
cor_df <- dplyr::filter(cor_df, abs_r >= .05) 
cor_df 
cal_words_ngrams_G4_CAL <- as.data.frame(cor_df[,3])

#### Random Forest  ####
df <- df [,colnames(df) %in% cal_words_ngrams_G4_CAL[,1]]
modeling_df <- cbind (outcome_df[,2], df)
colnames(modeling_df) <- paste(colnames(modeling_df), "_rf", sep = "")
colnames(modeling_df) <- gsub(" ", "__", colnames(modeling_df))
colnames(modeling_df)[1] <- 'Ratee_G4_num'
modeling_df <- as.data.frame (apply (modeling_df, 2, scale)) #normalizing entire dataset
modeling_df [1:5, 1:10]
dim (modeling_df)

num_cores <- detectCores() - 1 # Run in parallel across processors on local machine
cl.spec <- rep("localhost", num_cores) # Run in parallel across processors on local machine
cl <- parallel::makeCluster(cl.spec) # Run in parallel across processors on local machine
registerDoParallel(cl, cores=3)
set.seed(333333)
#run below to determine the optimal mtry, or number of variables per node
#Note: explored optimal number of trees in separate analyses, with anywhere from 2-3k being about the same performance
#fit  <- tuneRF(modeling_df[,2:ncol(modeling_df)], y=modeling_df[,1], mtryStart=(ncol(modeling_df)/5),ntreeTry=2000,stepFactor=1.25,improve=.00001) 
#fit
rf_model <-  randomForest (Ratee_G4_num ~ ., data = modeling_df, nodesize=5, ntree=2000, mtry= 16) #manually input the optimal number of variables from step above
rf_model
saveRDS(rf_model, "RF_fit_ngram_G4_overallCAL.Rda")
stopCluster(cl)
closeAllConnections()
registerDoSEQ()

######
###### G5 #####
#######
df <- readRDS ("G5_dtm_m.Rda")
dim (df)
outcome_df <- df [,1:3] #assumes first variable is ID, second variable is performance rating, and third variable is the sample indicator
df <- as.matrix (df [,4:ncol(df)]) #selecting the predictor variables
df <- as.data.frame (apply (df, 2, scale)) #normalizing
df [1:10, 1:10]
outcome_df [1:10,]

#### correlational filter (bootstrapping to reduce sample dependency) ####
set.seed(11111)
dat <- cbind(as.matrix(outcome_df[,2]), df)
N <- 1000
R <- 50 # running 50 bootstrapped samples and taking the average
cor_df  <-   data.frame(matrix(NA, nrow = ncol(df), ncol = R))
for (i in 1:R) {
  x <- dat [sample(nrow(dat),size=nrow(dat),replace=TRUE),]
  res <- corr.test(x[,2:ncol(x)], as.matrix(x[,1]))$r
  res [,1] [is.na(res[,1])] <- 0
  cor_df [, i] <- res [,1]
}
cor_df$r <- rowMeans(cor_df[,1:R])
cor_df <- dplyr::select (cor_df, r)
cor_df$abs_r <- abs(cor_df[,1])
cor_df <- cbind (cor_df, colnames(df))
cor_df <- dplyr::arrange(cor_df, desc(abs_r))
cor_df <- dplyr::filter(cor_df, abs_r >= .05) 
cor_df 
cal_words_ngrams_G5_CAL <- as.data.frame(cor_df[,3])

#### Random Forest  ####
df <- df [,colnames(df) %in% cal_words_ngrams_G5_CAL[,1]]
modeling_df <- cbind (outcome_df[,2], df)
colnames(modeling_df) <- paste(colnames(modeling_df), "_rf", sep = "")
colnames(modeling_df) <- gsub(" ", "__", colnames(modeling_df))
colnames(modeling_df)[1] <- 'Ratee_G5_num'
modeling_df <- as.data.frame (apply (modeling_df, 2, scale)) #normalizing entire dataset
modeling_df [1:5, 1:10]
dim (modeling_df)

num_cores <- detectCores() - 1 # Run in parallel across processors on local machine
cl.spec <- rep("localhost", num_cores) # Run in parallel across processors on local machine
cl <- parallel::makeCluster(cl.spec) # Run in parallel across processors on local machine
registerDoParallel(cl, cores=3)
set.seed(333333)
#run below to determine the optimal mtry, or number of variables per node
#Note: explored optimal number of trees in separate analyses, with anywhere from 2-3k being about the same performance
#fit  <- tuneRF(modeling_df[,2:ncol(modeling_df)], y=modeling_df[,1], mtryStart=(ncol(modeling_df)/5),ntreeTry=2000,stepFactor=1.25,improve=.00001) 
#fit
rf_model <-  randomForest (Ratee_G5_num ~ ., data = modeling_df, nodesize=5, ntree=2000, mtry= 50) #manually input the optimal number of variables from step above
rf_model
saveRDS(rf_model, "RF_fit_ngram_G5_overallCAL.Rda")
stopCluster(cl)
closeAllConnections()
registerDoSEQ()

######
###### G6 #####
#######
df <- readRDS ("G6_dtm_m.Rda")
dim (df)
outcome_df <- df [,1:3] #assumes first variable is ID, second variable is performance rating, and third variable is the sample indicator
df <- as.matrix (df [,4:ncol(df)]) #selecting the predictor variables
df <- as.data.frame (apply (df, 2, scale)) #normalizing
df [1:10, 1:10]
outcome_df [1:10,]

#### correlational filter (bootstrapping to reduce sample dependency) ####
set.seed(11111)
dat <- cbind(as.matrix(outcome_df[,2]), df)
N <- 1000
R <- 50 # running 50 bootstrapped samples and taking the average
cor_df  <-   data.frame(matrix(NA, nrow = ncol(df), ncol = R))
for (i in 1:R) {
  x <- dat [sample(nrow(dat),size=nrow(dat),replace=TRUE),]
  res <- corr.test(x[,2:ncol(x)], as.matrix(x[,1]))$r
  res [,1] [is.na(res[,1])] <- 0
  cor_df [, i] <- res [,1]
}
cor_df$r <- rowMeans(cor_df[,1:R])
cor_df <- dplyr::select (cor_df, r)
cor_df$abs_r <- abs(cor_df[,1])
cor_df <- cbind (cor_df, colnames(df))
cor_df <- dplyr::arrange(cor_df, desc(abs_r))
cor_df <- dplyr::filter(cor_df, abs_r >= .05) 
cor_df 
cal_words_ngrams_G6_CAL <- as.data.frame(cor_df[,3])

#### Random Forest  ####
df <- df [,colnames(df) %in% cal_words_ngrams_G6_CAL[,1]]
modeling_df <- cbind (outcome_df[,2], df)
colnames(modeling_df) <- paste(colnames(modeling_df), "_rf", sep = "")
colnames(modeling_df) <- gsub(" ", "__", colnames(modeling_df))
colnames(modeling_df)[1] <- 'Ratee_G6_num'
modeling_df <- as.data.frame (apply (modeling_df, 2, scale)) #normalizing entire dataset
modeling_df [1:5, 1:10]
dim (modeling_df)

num_cores <- detectCores() - 1 # Run in parallel across processors on local machine
cl.spec <- rep("localhost", num_cores) # Run in parallel across processors on local machine
cl <- parallel::makeCluster(cl.spec) # Run in parallel across processors on local machine
registerDoParallel(cl, cores=3)
set.seed(333333)
#run below to determine the optimal mtry, or number of variables per node
#Note: explored optimal number of trees in separate analyses, with anywhere from 2-3k being about the same performance
#fit  <- tuneRF(modeling_df[,2:ncol(modeling_df)], y=modeling_df[,1], mtryStart=(ncol(modeling_df)/5),ntreeTry=2000,stepFactor=1.25,improve=.00001) 
#fit
rf_model <-  randomForest (Ratee_G6_num ~ ., data = modeling_df, nodesize=5, ntree=2000, mtry= 28) #manually input the optimal number of variables from step above
rf_model
saveRDS(rf_model, "RF_fit_ngram_G6_overallCAL.Rda")
stopCluster(cl)
closeAllConnections()
registerDoSEQ()

######
###### G7 #####
#######
df <- readRDS ("G7_dtm_m.Rda")
dim (df)
outcome_df <- df [,1:3] #assumes first variable is ID, second variable is performance rating, and third variable is the sample indicator
df <- as.matrix (df [,4:ncol(df)]) #selecting the predictor variables
df <- as.data.frame (apply (df, 2, scale)) #normalizing
df [1:10, 1:10]
outcome_df [1:10,]

#### correlational filter (bootstrapping to reduce sample dependency) ####
set.seed(11111)
dat <- cbind(as.matrix(outcome_df[,2]), df)
N <- 1000
R <- 50 # running 50 bootstrapped samples and taking the average
cor_df  <-   data.frame(matrix(NA, nrow = ncol(df), ncol = R))
for (i in 1:R) {
  x <- dat [sample(nrow(dat),size=nrow(dat),replace=TRUE),]
  res <- corr.test(x[,2:ncol(x)], as.matrix(x[,1]))$r
  res [,1] [is.na(res[,1])] <- 0
  cor_df [, i] <- res [,1]
}
cor_df$r <- rowMeans(cor_df[,1:R])
cor_df <- dplyr::select (cor_df, r)
cor_df$abs_r <- abs(cor_df[,1])
cor_df <- cbind (cor_df, colnames(df))
cor_df <- dplyr::arrange(cor_df, desc(abs_r))
cor_df <- dplyr::filter(cor_df, abs_r >= .05) 
cor_df 
cal_words_ngrams_G7_CAL <- as.data.frame(cor_df[,3])

#### Random Forest  ####
df <- df [,colnames(df) %in% cal_words_ngrams_G7_CAL[,1]]
modeling_df <- cbind (outcome_df[,2], df)
colnames(modeling_df) <- paste(colnames(modeling_df), "_rf", sep = "")
colnames(modeling_df) <- gsub(" ", "__", colnames(modeling_df))
colnames(modeling_df)[1] <- 'Ratee_G7_num'
modeling_df <- as.data.frame (apply (modeling_df, 2, scale)) #normalizing entire dataset
modeling_df [1:5, 1:10]
dim (modeling_df)

num_cores <- detectCores() - 1 # Run in parallel across processors on local machine
cl.spec <- rep("localhost", num_cores) # Run in parallel across processors on local machine
cl <- parallel::makeCluster(cl.spec) # Run in parallel across processors on local machine
registerDoParallel(cl, cores=3)
set.seed(333333)
#run below to determine the optimal mtry, or number of variables per node
#Note: explored optimal number of trees in separate analyses, with anywhere from 2-3k being about the same performance
#fit  <- tuneRF(modeling_df[,2:ncol(modeling_df)], y=modeling_df[,1], mtryStart=(ncol(modeling_df)/5),ntreeTry=2000,stepFactor=1.25,improve=.00001) 
#fit
rf_model <-  randomForest (Ratee_G7_num ~ ., data = modeling_df, nodesize=5, ntree=2000, mtry= 10) #manually input the optimal number of variables from step above
rf_model
saveRDS(rf_model, "RF_fit_ngram_G7_overallCAL.Rda")
stopCluster(cl)
closeAllConnections()
registerDoSEQ()

######
###### G8 #####
#######
df <- readRDS ("G8_dtm_m.Rda")
dim (df)
outcome_df <- df [,1:3] #assumes first variable is ID, second variable is performance rating, and third variable is the sample indicator
df <- as.matrix (df [,4:ncol(df)]) #selecting the predictor variables
df <- as.data.frame (apply (df, 2, scale)) #normalizing
df [1:10, 1:10]
outcome_df [1:10,]

#### correlational filter (bootstrapping to reduce sample dependency) ####
set.seed(11111)
dat <- cbind(as.matrix(outcome_df[,2]), df)
N <- 1000
R <- 50 # running 50 bootstrapped samples and taking the average
cor_df  <-   data.frame(matrix(NA, nrow = ncol(df), ncol = R))
for (i in 1:R) {
  x <- dat [sample(nrow(dat),size=nrow(dat),replace=TRUE),]
  res <- corr.test(x[,2:ncol(x)], as.matrix(x[,1]))$r
  res [,1] [is.na(res[,1])] <- 0
  cor_df [, i] <- res [,1]
}
cor_df$r <- rowMeans(cor_df[,1:R])
cor_df <- dplyr::select (cor_df, r)
cor_df$abs_r <- abs(cor_df[,1])
cor_df <- cbind (cor_df, colnames(df))
cor_df <- dplyr::arrange(cor_df, desc(abs_r))
cor_df <- dplyr::filter(cor_df, abs_r >= .05) 
cor_df 
cal_words_ngrams_G8_CAL <- as.data.frame(cor_df[,3])

#### Random Forest  ####
df <- df [,colnames(df) %in% cal_words_ngrams_G8_CAL[,1]]
modeling_df <- cbind (outcome_df[,2], df)
colnames(modeling_df) <- paste(colnames(modeling_df), "_rf", sep = "")
colnames(modeling_df) <- gsub(" ", "__", colnames(modeling_df))
colnames(modeling_df)[1] <- 'Ratee_G8_num'
modeling_df <- as.data.frame (apply (modeling_df, 2, scale)) #normalizing entire dataset
modeling_df [1:5, 1:10]
dim (modeling_df)

num_cores <- detectCores() - 1 # Run in parallel across processors on local machine
cl.spec <- rep("localhost", num_cores) # Run in parallel across processors on local machine
cl <- parallel::makeCluster(cl.spec) # Run in parallel across processors on local machine
registerDoParallel(cl, cores=3)
set.seed(333333)
#run below to determine the optimal mtry, or number of variables per node
#Note: explored optimal number of trees in separate analyses, with anywhere from 2-3k being about the same performance
#fit  <- tuneRF(modeling_df[,2:ncol(modeling_df)], y=modeling_df[,1], mtryStart=(ncol(modeling_df)/5),ntreeTry=2000,stepFactor=1.25,improve=.00001) 
#fit
rf_model <-  randomForest (Ratee_G8_num ~ ., data = modeling_df, nodesize=5, ntree=2000, mtry= 24) #manually input the optimal number of variables from step above
rf_model
saveRDS(rf_model, "RF_fit_ngram_G8_overallCAL.Rda")
stopCluster(cl)
closeAllConnections()
registerDoSEQ()

######
###### Overall N-Grams #####
#######
df <- readRDS ("Overall_dtm_m.Rda")
dim (df)
outcome_df <- df [,1:3] #assumes first variable is ID, second variable is performance rating, and third variable is the sample indicator
df <- as.matrix (df [,4:ncol(df)]) #selecting the predictor variables
df <- as.data.frame (apply (df, 2, scale)) #normalizing
df [1:10, 1:10]
outcome_df [1:10,]

#### correlational filter (bootstrapping to reduce sample dependency) ####
set.seed(11111)
dat <- cbind(as.matrix(outcome_df[,2]), df)
N <- 1000
R <- 50 # running 50 bootstrapped samples and taking the average
cor_df  <-   data.frame(matrix(NA, nrow = ncol(df), ncol = R))
for (i in 1:R) {
  x <- dat [sample(nrow(dat),size=nrow(dat),replace=TRUE),]
  res <- corr.test(x[,2:ncol(x)], as.matrix(x[,1]))$r
  res [,1] [is.na(res[,1])] <- 0
  cor_df [, i] <- res [,1]
}
cor_df$r <- rowMeans(cor_df[,1:R])
cor_df <- dplyr::select (cor_df, r)
cor_df$abs_r <- abs(cor_df[,1])
cor_df <- cbind (cor_df, colnames(df))
cor_df <- dplyr::arrange(cor_df, desc(abs_r))
cor_df <- dplyr::filter(cor_df, abs_r >= .05) 
cor_df 
cal_words_ngrams_overall_CAL <- as.data.frame(cor_df[,3])

#### Random Forest  ####
df <- df [,colnames(df) %in% cal_words_ngrams_overall_CAL[,1]]
modeling_df <- cbind (outcome_df[,2], df)
colnames(modeling_df) <- paste(colnames(modeling_df), "_rf", sep = "")
colnames(modeling_df) <- gsub(" ", "__", colnames(modeling_df))
colnames(modeling_df)[1] <- 'Performance_Composite'
modeling_df <- as.data.frame (apply (modeling_df, 2, scale)) #normalizing entire dataset
modeling_df [1:5, 1:10]
dim (modeling_df)

num_cores <- detectCores() - 1 # Run in parallel across processors on local machine
cl.spec <- rep("localhost", num_cores) # Run in parallel across processors on local machine
cl <- parallel::makeCluster(cl.spec) # Run in parallel across processors on local machine
registerDoParallel(cl, cores=3)
set.seed(333333)
#run below to determine the optimal mtry, or number of variables per node
#Note: explored optimal number of trees in separate analyses, with anywhere from 2-3k being about the same performance
#fit  <- tuneRF(modeling_df[,2:ncol(modeling_df)], y=modeling_df[,1], mtryStart=(ncol(modeling_df)/5),ntreeTry=2000,stepFactor=1.25,improve=.00001) 
#fit
rf_model <-  randomForest (Performance_Composite ~ ., data = modeling_df, nodesize=5, ntree=2000, mtry= 187) #manually input the optimal number of variables from step above
rf_model
saveRDS(rf_model, "RF_fit_ngram_overall_overallCAL.Rda")
stopCluster(cl)
closeAllConnections()
registerDoSEQ()

###
### Making Word List for all Great 8 Dimenions ###
###
cal_words_ngrams_G1_CAL$Dimension <- 'G1'
cal_words_ngrams_G2_CAL$Dimension <- 'G2'
cal_words_ngrams_G3_CAL$Dimension <- 'G3'
cal_words_ngrams_G4_CAL$Dimension <- 'G4'
cal_words_ngrams_G5_CAL$Dimension <- 'G5'
cal_words_ngrams_G6_CAL$Dimension <- 'G6'
cal_words_ngrams_G7_CAL$Dimension <- 'G7'
cal_words_ngrams_G8_CAL$Dimension <- 'G8'
cal_words_ngrams_overall_CAL$Dimension <- 'overall'
cal_words_ngrams_CAL <- rbind (cal_words_ngrams_G1_CAL, cal_words_ngrams_G2_CAL, cal_words_ngrams_G3_CAL, cal_words_ngrams_G4_CAL, cal_words_ngrams_G5_CAL, 
                               cal_words_ngrams_G6_CAL, cal_words_ngrams_G7_CAL, cal_words_ngrams_G8_CAL, cal_words_ngrams_overall_CAL)
colnames(cal_words_ngrams_CAL)[1] <- 'word'
nrow (cal_words_ngrams_CAL)
saveRDS(cal_words_ngrams_CAL, "cal_words_ngrams_CAL.Rda")

############################################################################################################################################
############################################################################################################################################
#################################################   Building traditional BOW (full dtm) Models  ###################################################
############################################################################################################################################
############################################################################################################################################

######
###### G1 #####
#######
Perf <- dplyr::select(Text_df, Ratee_G1_num)
df <- readRDS ("Overall_dtm_m.Rda")
outcome_df <- cbind (  dplyr::select(df, Unique_ID_Ratee), Perf, dplyr::select(df, Train_filter) )
df <- as.matrix (df [,4:ncol(df)]) #selecting the predictor variables
df <- as.data.frame (apply (df, 2, scale)) #normalizing
df [1:10, 1:10]
outcome_df [1:10,]

#### correlational filter (bootstrapping to reduce sample dependency) ####
set.seed(11111)
dat <- cbind(as.matrix(outcome_df[,2]), df)
N <- 1000
R <- 50 # running 50 bootstrapped samples and taking the average
cor_df  <-   data.frame(matrix(NA, nrow = ncol(df), ncol = R))
for (i in 1:R) {
  x <- dat [sample(nrow(dat),size=nrow(dat),replace=TRUE),]
  res <- corr.test(x[,2:ncol(x)], as.matrix(x[,1]))$r
  res [,1] [is.na(res[,1])] <- 0
  cor_df [, i] <- res [,1]
}
cor_df$r <- rowMeans(cor_df[,1:R])
cor_df <- dplyr::select (cor_df, r)
cor_df$abs_r <- abs(cor_df[,1])
cor_df <- cbind (cor_df, colnames(df))
cor_df <- dplyr::arrange(cor_df, desc(abs_r))
cor_df <- dplyr::filter(cor_df, abs_r >= .05) 
cor_df 
cal_words_fulldtm_G1_CAL <- as.data.frame(cor_df[,3])

#### Random Forest  ####
df <- df [,colnames(df) %in% cal_words_fulldtm_G1_CAL[,1]]
modeling_df <- cbind (outcome_df[,2], df)
colnames(modeling_df) <- paste(colnames(modeling_df), "_rf", sep = "")
colnames(modeling_df) <- gsub(" ", "__", colnames(modeling_df))
colnames(modeling_df)[1] <- 'Ratee_G1_num'
modeling_df <- as.data.frame (apply (modeling_df, 2, scale)) #normalizing entire dataset
modeling_df [1:5, 1:10]
dim (modeling_df)

num_cores <- detectCores() - 1 # Run in parallel across processors on local machine
cl.spec <- rep("localhost", num_cores) # Run in parallel across processors on local machine
cl <- parallel::makeCluster(cl.spec) # Run in parallel across processors on local machine
registerDoParallel(cl, cores=3)
set.seed(333333)
#run below to determine the optimal mtry, or number of variables per node
#Note: explored optimal number of trees in separate analyses, with anywhere from 2-3k being about the same performance
#fit  <- tuneRF(modeling_df[,2:ncol(modeling_df)], y=modeling_df[,1], mtryStart=(ncol(modeling_df)/5),ntreeTry=2000,stepFactor=1.25,improve=.00001) 
#fit
rf_model <-  randomForest (Ratee_G1_num ~ ., data = modeling_df, nodesize=5, ntree=2000, mtry= 178) #manually input the optimal number of variables from step above
rf_model
saveRDS(rf_model, "RF_fit_fulldtm_G1_overallCAL.Rda")
stopCluster(cl)
closeAllConnections()
registerDoSEQ()

######
###### G2 #####
#######
Perf <- dplyr::select(Text_df, Ratee_G2_num)
df <- readRDS ("Overall_dtm_m.Rda")
outcome_df <- cbind (  dplyr::select(df, Unique_ID_Ratee), Perf, dplyr::select(df, Train_filter) )
df <- as.matrix (df [,4:ncol(df)]) #selecting the predictor variables
df <- as.data.frame (apply (df, 2, scale)) #normalizing
df [1:10, 1:10]
outcome_df [1:10,]

#### correlational filter (bootstrapping to reduce sample dependency) ####
set.seed(11111)
dat <- cbind(as.matrix(outcome_df[,2]), df)
N <- 1000
R <- 50 # running 50 bootstrapped samples and taking the average
cor_df  <-   data.frame(matrix(NA, nrow = ncol(df), ncol = R))
for (i in 1:R) {
  x <- dat [sample(nrow(dat),size=nrow(dat),replace=TRUE),]
  res <- corr.test(x[,2:ncol(x)], as.matrix(x[,1]))$r
  res [,1] [is.na(res[,1])] <- 0
  cor_df [, i] <- res [,1]
}
cor_df$r <- rowMeans(cor_df[,1:R])
cor_df <- dplyr::select (cor_df, r)
cor_df$abs_r <- abs(cor_df[,1])
cor_df <- cbind (cor_df, colnames(df))
cor_df <- dplyr::arrange(cor_df, desc(abs_r))
cor_df <- dplyr::filter(cor_df, abs_r >= .05) 
cor_df 
cal_words_fulldtm_G2_CAL <- as.data.frame(cor_df[,3])

#### Random Forest  ####
df <- df [,colnames(df) %in% cal_words_fulldtm_G2_CAL[,1]]
modeling_df <- cbind (outcome_df[,2], df)
colnames(modeling_df) <- paste(colnames(modeling_df), "_rf", sep = "")
colnames(modeling_df) <- gsub(" ", "__", colnames(modeling_df))
colnames(modeling_df)[1] <- 'Ratee_G2_num'
modeling_df <- modeling_df [,(grepl("doesnâ", colnames(modeling_df))==FALSE)]
modeling_df <- as.data.frame (apply (modeling_df, 2, scale)) #normalizing entire dataset
modeling_df [1:5, 1:10]
dim (modeling_df)

num_cores <- detectCores() - 1 # Run in parallel across processors on local machine
cl.spec <- rep("localhost", num_cores) # Run in parallel across processors on local machine
cl <- parallel::makeCluster(cl.spec) # Run in parallel across processors on local machine
registerDoParallel(cl, cores=3)
set.seed(333333)
#run below to determine the optimal mtry, or number of variables per node
#Note: explored optimal number of trees in separate analyses, with anywhere from 2-3k being about the same performance
#fit  <- tuneRF(modeling_df[,2:ncol(modeling_df)], y=modeling_df[,1], mtryStart=(ncol(modeling_df)/5),ntreeTry=2000,stepFactor=1.25,improve=.00001) 
#fit
rf_model <-  randomForest (Ratee_G2_num ~ ., data = modeling_df, nodesize=5, ntree=2000, mtry= 193) #manually input the optimal number of variables from step above
rf_model
saveRDS(rf_model, "RF_fit_fulldtm_G2_overallCAL.Rda")
stopCluster(cl)
closeAllConnections()
registerDoSEQ()

######
###### G3 #####
#######
Perf <- dplyr::select(Text_df, Ratee_G3_num)
df <- readRDS ("Overall_dtm_m.Rda")
outcome_df <- cbind (  dplyr::select(df, Unique_ID_Ratee), Perf, dplyr::select(df, Train_filter) )
df <- as.matrix (df [,4:ncol(df)]) #selecting the predictor variables
df <- as.data.frame (apply (df, 2, scale)) #normalizing
df [1:10, 1:10]
outcome_df [1:10,]

#### correlational filter (bootstrapping to reduce sample dependency) ####
set.seed(11111)
dat <- cbind(as.matrix(outcome_df[,2]), df)
N <- 1000
R <- 50 # running 50 bootstrapped samples and taking the average
cor_df  <-   data.frame(matrix(NA, nrow = ncol(df), ncol = R))
for (i in 1:R) {
  x <- dat [sample(nrow(dat),size=nrow(dat),replace=TRUE),]
  res <- corr.test(x[,2:ncol(x)], as.matrix(x[,1]))$r
  res [,1] [is.na(res[,1])] <- 0
  cor_df [, i] <- res [,1]
}
cor_df$r <- rowMeans(cor_df[,1:R])
cor_df <- dplyr::select (cor_df, r)
cor_df$abs_r <- abs(cor_df[,1])
cor_df <- cbind (cor_df, colnames(df))
cor_df <- dplyr::arrange(cor_df, desc(abs_r))
cor_df <- dplyr::filter(cor_df, abs_r >= .05) 
cor_df 
cal_words_fulldtm_G3_CAL <- as.data.frame(cor_df[,3])

#### Random Forest  ####
df <- df [,colnames(df) %in% cal_words_fulldtm_G3_CAL[,1]]
modeling_df <- cbind (outcome_df[,2], df)
colnames(modeling_df) <- paste(colnames(modeling_df), "_rf", sep = "")
colnames(modeling_df) <- gsub(" ", "__", colnames(modeling_df))
colnames(modeling_df)[1] <- 'Ratee_G3_num'
modeling_df <- as.data.frame (apply (modeling_df, 2, scale)) #normalizing entire dataset
modeling_df [1:5, 1:10]
dim (modeling_df)

num_cores <- detectCores() - 1 # Run in parallel across processors on local machine
cl.spec <- rep("localhost", num_cores) # Run in parallel across processors on local machine
cl <- parallel::makeCluster(cl.spec) # Run in parallel across processors on local machine
registerDoParallel(cl, cores=3)
set.seed(333333)
#run below to determine the optimal mtry, or number of variables per node
#Note: explored optimal number of trees in separate analyses, with anywhere from 2-3k being about the same performance
#fit  <- tuneRF(modeling_df[,2:ncol(modeling_df)], y=modeling_df[,1], mtryStart=(ncol(modeling_df)/5),ntreeTry=2000,stepFactor=1.25,improve=.00001) 
#fit
rf_model <-  randomForest (Ratee_G3_num ~ ., data = modeling_df, nodesize=5, ntree=2000, mtry= 166) #manually input the optimal number of variables from step above
rf_model
saveRDS(rf_model, "RF_fit_fulldtm_G3_overallCAL.Rda")
stopCluster(cl)
closeAllConnections()
registerDoSEQ()

######
###### G4 #####
#######
Perf <- dplyr::select(Text_df, Ratee_G4_num)
df <- readRDS ("Overall_dtm_m.Rda")
outcome_df <- cbind (  dplyr::select(df, Unique_ID_Ratee), Perf, dplyr::select(df, Train_filter) )
df <- as.matrix (df [,4:ncol(df)]) #selecting the predictor variables
df <- as.data.frame (apply (df, 2, scale)) #normalizing
df [1:10, 1:10]
outcome_df [1:10,]

#### correlational filter (bootstrapping to reduce sample dependency) ####
set.seed(11111)
dat <- cbind(as.matrix(outcome_df[,2]), df)
N <- 1000
R <- 50 # running 50 bootstrapped samples and taking the average
cor_df  <-   data.frame(matrix(NA, nrow = ncol(df), ncol = R))
for (i in 1:R) {
  x <- dat [sample(nrow(dat),size=nrow(dat),replace=TRUE),]
  res <- corr.test(x[,2:ncol(x)], as.matrix(x[,1]))$r
  res [,1] [is.na(res[,1])] <- 0
  cor_df [, i] <- res [,1]
}
cor_df$r <- rowMeans(cor_df[,1:R])
cor_df <- dplyr::select (cor_df, r)
cor_df$abs_r <- abs(cor_df[,1])
cor_df <- cbind (cor_df, colnames(df))
cor_df <- dplyr::arrange(cor_df, desc(abs_r))
cor_df <- dplyr::filter(cor_df, abs_r >= .05) 
cor_df 
cal_words_fulldtm_G4_CAL <- as.data.frame(cor_df[,3])

#### Random Forest  ####
df <- df [,colnames(df) %in% cal_words_fulldtm_G4_CAL[,1]]
modeling_df <- cbind (outcome_df[,2], df)
colnames(modeling_df) <- paste(colnames(modeling_df), "_rf", sep = "")
colnames(modeling_df) <- gsub(" ", "__", colnames(modeling_df))
colnames(modeling_df)[1] <- 'Ratee_G4_num'
modeling_df <- as.data.frame (apply (modeling_df, 2, scale)) #normalizing entire dataset
modeling_df [1:5, 1:10]
dim (modeling_df)

num_cores <- detectCores() - 1 # Run in parallel across processors on local machine
cl.spec <- rep("localhost", num_cores) # Run in parallel across processors on local machine
cl <- parallel::makeCluster(cl.spec) # Run in parallel across processors on local machine
registerDoParallel(cl, cores=3)
set.seed(333333)
#run below to determine the optimal mtry, or number of variables per node
#Note: explored optimal number of trees in separate analyses, with anywhere from 2-3k being about the same performance
#fit  <- tuneRF(modeling_df[,2:ncol(modeling_df)], y=modeling_df[,1], mtryStart=(ncol(modeling_df)/5),ntreeTry=2000,stepFactor=1.25,improve=.00001) 
#fit
rf_model <-  randomForest (Ratee_G4_num ~ ., data = modeling_df, nodesize=5, ntree=2000, mtry= 158) #manually input the optimal number of variables from step above
rf_model
saveRDS(rf_model, "RF_fit_fulldtm_G4_overallCAL.Rda")
stopCluster(cl)
closeAllConnections()
registerDoSEQ()

######
###### G5 #####
#######
Perf <- dplyr::select(Text_df, Ratee_G5_num)
df <- readRDS ("Overall_dtm_m.Rda")
outcome_df <- cbind (  dplyr::select(df, Unique_ID_Ratee), Perf, dplyr::select(df, Train_filter) )
df <- as.matrix (df [,4:ncol(df)]) #selecting the predictor variables
df <- as.data.frame (apply (df, 2, scale)) #normalizing
df [1:10, 1:10]
outcome_df [1:10,]

#### correlational filter (bootstrapping to reduce sample dependency) ####
set.seed(11111)
dat <- cbind(as.matrix(outcome_df[,2]), df)
N <- 1000
R <- 50 # running 50 bootstrapped samples and taking the average
cor_df  <-   data.frame(matrix(NA, nrow = ncol(df), ncol = R))
for (i in 1:R) {
  x <- dat [sample(nrow(dat),size=nrow(dat),replace=TRUE),]
  res <- corr.test(x[,2:ncol(x)], as.matrix(x[,1]))$r
  res [,1] [is.na(res[,1])] <- 0
  cor_df [, i] <- res [,1]
}
cor_df$r <- rowMeans(cor_df[,1:R])
cor_df <- dplyr::select (cor_df, r)
cor_df$abs_r <- abs(cor_df[,1])
cor_df <- cbind (cor_df, colnames(df))
cor_df <- dplyr::arrange(cor_df, desc(abs_r))
cor_df <- dplyr::filter(cor_df, abs_r >= .05) 
cor_df 
cal_words_fulldtm_G5_CAL <- as.data.frame(cor_df[,3])

#### Random Forest  ####
df <- df [,colnames(df) %in% cal_words_fulldtm_G5_CAL[,1]]
modeling_df <- cbind (outcome_df[,2], df)
colnames(modeling_df) <- paste(colnames(modeling_df), "_rf", sep = "")
colnames(modeling_df) <- gsub(" ", "__", colnames(modeling_df))
colnames(modeling_df)[1] <- 'Ratee_G5_num'
modeling_df <- as.data.frame (apply (modeling_df, 2, scale)) #normalizing entire dataset
modeling_df [1:5, 1:10]
dim (modeling_df)

num_cores <- detectCores() - 1 # Run in parallel across processors on local machine
cl.spec <- rep("localhost", num_cores) # Run in parallel across processors on local machine
cl <- parallel::makeCluster(cl.spec) # Run in parallel across processors on local machine
registerDoParallel(cl, cores=3)
set.seed(333333)
#run below to determine the optimal mtry, or number of variables per node
#Note: explored optimal number of trees in separate analyses, with anywhere from 2-3k being about the same performance
#fit  <- tuneRF(modeling_df[,2:ncol(modeling_df)], y=modeling_df[,1], mtryStart=(ncol(modeling_df)/5),ntreeTry=2000,stepFactor=1.25,improve=.00001) 
#fit
rf_model <-  randomForest (Ratee_G5_num ~ ., data = modeling_df, nodesize=5, ntree=2000, mtry= 103) #manually input the optimal number of variables from step above
rf_model
saveRDS(rf_model, "RF_fit_fulldtm_G5_overallCAL.Rda")
stopCluster(cl)
closeAllConnections()
registerDoSEQ()

######
###### G6 #####
#######
Perf <- dplyr::select(Text_df, Ratee_G6_num)
df <- readRDS ("Overall_dtm_m.Rda")
outcome_df <- cbind (  dplyr::select(df, Unique_ID_Ratee), Perf, dplyr::select(df, Train_filter) )
df <- as.matrix (df [,4:ncol(df)]) #selecting the predictor variables
df <- as.data.frame (apply (df, 2, scale)) #normalizing
df [1:10, 1:10]
outcome_df [1:10,]

#### correlational filter (bootstrapping to reduce sample dependency) ####
set.seed(11111)
dat <- cbind(as.matrix(outcome_df[,2]), df)
N <- 1000
R <- 50 # running 50 bootstrapped samples and taking the average
cor_df  <-   data.frame(matrix(NA, nrow = ncol(df), ncol = R))
for (i in 1:R) {
  x <- dat [sample(nrow(dat),size=nrow(dat),replace=TRUE),]
  res <- corr.test(x[,2:ncol(x)], as.matrix(x[,1]))$r
  res [,1] [is.na(res[,1])] <- 0
  cor_df [, i] <- res [,1]
}
cor_df$r <- rowMeans(cor_df[,1:R])
cor_df <- dplyr::select (cor_df, r)
cor_df$abs_r <- abs(cor_df[,1])
cor_df <- cbind (cor_df, colnames(df))
cor_df <- dplyr::arrange(cor_df, desc(abs_r))
cor_df <- dplyr::filter(cor_df, abs_r >= .05) 
cor_df 
cal_words_fulldtm_G6_CAL <- as.data.frame(cor_df[,3])

#### Random Forest  ####
df <- df [,colnames(df) %in% cal_words_fulldtm_G6_CAL[,1]]
modeling_df <- cbind (outcome_df[,2], df)
colnames(modeling_df) <- paste(colnames(modeling_df), "_rf", sep = "")
colnames(modeling_df) <- gsub(" ", "__", colnames(modeling_df))
colnames(modeling_df)[1] <- 'Ratee_G6_num'
modeling_df <- as.data.frame (apply (modeling_df, 2, scale)) #normalizing entire dataset
modeling_df [1:5, 1:10]
dim (modeling_df)

num_cores <- detectCores() - 1 # Run in parallel across processors on local machine
cl.spec <- rep("localhost", num_cores) # Run in parallel across processors on local machine
cl <- parallel::makeCluster(cl.spec) # Run in parallel across processors on local machine
registerDoParallel(cl, cores=3)
set.seed(333333)
#run below to determine the optimal mtry, or number of variables per node
#Note: explored optimal number of trees in separate analyses, with anywhere from 2-3k being about the same performance
#fit  <- tuneRF(modeling_df[,2:ncol(modeling_df)], y=modeling_df[,1], mtryStart=(ncol(modeling_df)/5),ntreeTry=2000,stepFactor=1.25,improve=.00001) 
#fit
rf_model <-  randomForest (Ratee_G6_num ~ ., data = modeling_df, nodesize=5, ntree=2000, mtry= 219) #manually input the optimal number of variables from step above
rf_model
saveRDS(rf_model, "RF_fit_fulldtm_G6_overallCAL.Rda")
stopCluster(cl)
closeAllConnections()
registerDoSEQ()

######
###### G7 #####
#######
Perf <- dplyr::select(Text_df, Ratee_G7_num)
df <- readRDS ("Overall_dtm_m.Rda")
outcome_df <- cbind (  dplyr::select(df, Unique_ID_Ratee), Perf, dplyr::select(df, Train_filter) )
df <- as.matrix (df [,4:ncol(df)]) #selecting the predictor variables
df <- as.data.frame (apply (df, 2, scale)) #normalizing
df [1:10, 1:10]
outcome_df [1:10,]

#### correlational filter (bootstrapping to reduce sample dependency) ####
set.seed(11111)
dat <- cbind(as.matrix(outcome_df[,2]), df)
N <- 1000
R <- 50 # running 50 bootstrapped samples and taking the average
cor_df  <-   data.frame(matrix(NA, nrow = ncol(df), ncol = R))
for (i in 1:R) {
  x <- dat [sample(nrow(dat),size=nrow(dat),replace=TRUE),]
  res <- corr.test(x[,2:ncol(x)], as.matrix(x[,1]))$r
  res [,1] [is.na(res[,1])] <- 0
  cor_df [, i] <- res [,1]
}
cor_df$r <- rowMeans(cor_df[,1:R])
cor_df <- dplyr::select (cor_df, r)
cor_df$abs_r <- abs(cor_df[,1])
cor_df <- cbind (cor_df, colnames(df))
cor_df <- dplyr::arrange(cor_df, desc(abs_r))
cor_df <- dplyr::filter(cor_df, abs_r >= .05) 
cor_df 
cal_words_fulldtm_G7_CAL <- as.data.frame(cor_df[,3])

#### Random Forest  ####
df <- df [,colnames(df) %in% cal_words_fulldtm_G7_CAL[,1]]
modeling_df <- cbind (outcome_df[,2], df)
colnames(modeling_df) <- paste(colnames(modeling_df), "_rf", sep = "")
colnames(modeling_df) <- gsub(" ", "__", colnames(modeling_df))
colnames(modeling_df)[1] <- 'Ratee_G7_num'
modeling_df <- as.data.frame (apply (modeling_df, 2, scale)) #normalizing entire dataset
modeling_df [1:5, 1:10]
dim (modeling_df)

num_cores <- detectCores() - 1 # Run in parallel across processors on local machine
cl.spec <- rep("localhost", num_cores) # Run in parallel across processors on local machine
cl <- parallel::makeCluster(cl.spec) # Run in parallel across processors on local machine
registerDoParallel(cl, cores=3)
set.seed(333333)
#run below to determine the optimal mtry, or number of variables per node
#Note: explored optimal number of trees in separate analyses, with anywhere from 2-3k being about the same performance
#fit  <- tuneRF(modeling_df[,2:ncol(modeling_df)], y=modeling_df[,1], mtryStart=(ncol(modeling_df)/5),ntreeTry=2000,stepFactor=1.25,improve=.00001) 
#fit
rf_model <-  randomForest (Ratee_G7_num ~ ., data = modeling_df, nodesize=5, ntree=2000, mtry= 170) #manually input the optimal number of variables from step above
rf_model
saveRDS(rf_model, "RF_fit_fulldtm_G7_overallCAL.Rda")
stopCluster(cl)
closeAllConnections()
registerDoSEQ()

######
###### G8 #####
#######
Perf <- dplyr::select(Text_df, Ratee_G8_num)
df <- readRDS ("Overall_dtm_m.Rda")
outcome_df <- cbind (  dplyr::select(df, Unique_ID_Ratee), Perf, dplyr::select(df, Train_filter) )
df <- as.matrix (df [,4:ncol(df)]) #selecting the predictor variables
df <- as.data.frame (apply (df, 2, scale)) #normalizing
df [1:10, 1:10]
outcome_df [1:10,]

#### correlational filter (bootstrapping to reduce sample dependency) ####
set.seed(11111)
dat <- cbind(as.matrix(outcome_df[,2]), df)
N <- 1000
R <- 50 # running 50 bootstrapped samples and taking the average
cor_df  <-   data.frame(matrix(NA, nrow = ncol(df), ncol = R))
for (i in 1:R) {
  x <- dat [sample(nrow(dat),size=nrow(dat),replace=TRUE),]
  res <- corr.test(x[,2:ncol(x)], as.matrix(x[,1]))$r
  res [,1] [is.na(res[,1])] <- 0
  cor_df [, i] <- res [,1]
}
cor_df$r <- rowMeans(cor_df[,1:R])
cor_df <- dplyr::select (cor_df, r)
cor_df$abs_r <- abs(cor_df[,1])
cor_df <- cbind (cor_df, colnames(df))
cor_df <- dplyr::arrange(cor_df, desc(abs_r))
cor_df <- dplyr::filter(cor_df, abs_r >= .05) 
cor_df 
cal_words_fulldtm_G8_CAL <- as.data.frame(cor_df[,3])

#### Random Forest  ####
df <- df [,colnames(df) %in% cal_words_fulldtm_G8_CAL[,1]]
modeling_df <- cbind (outcome_df[,2], df)
colnames(modeling_df) <- paste(colnames(modeling_df), "_rf", sep = "")
colnames(modeling_df) <- gsub(" ", "__", colnames(modeling_df))
colnames(modeling_df)[1] <- 'Ratee_G8_num'
modeling_df <- as.data.frame (apply (modeling_df, 2, scale)) #normalizing entire dataset
modeling_df [1:5, 1:10]
dim (modeling_df)

num_cores <- detectCores() - 1 # Run in parallel across processors on local machine
cl.spec <- rep("localhost", num_cores) # Run in parallel across processors on local machine
cl <- parallel::makeCluster(cl.spec) # Run in parallel across processors on local machine
registerDoParallel(cl, cores=3)
set.seed(333333)
#run below to determine the optimal mtry, or number of variables per node
#Note: explored optimal number of trees in separate analyses, with anywhere from 2-3k being about the same performance
#fit  <- tuneRF(modeling_df[,2:ncol(modeling_df)], y=modeling_df[,1], mtryStart=(ncol(modeling_df)/5),ntreeTry=2000,stepFactor=1.25,improve=.00001) 
#fit
rf_model <-  randomForest (Ratee_G8_num ~ ., data = modeling_df, nodesize=5, ntree=2000, mtry= 127) #manually input the optimal number of variables from step above
rf_model
saveRDS(rf_model, "RF_fit_fulldtm_G8_overallCAL.Rda")
stopCluster(cl)
closeAllConnections()
registerDoSEQ()

######
###### Overall #####
#######
RF_fit_fulldtm_overall_overallCA <- readRDS ("RF_fit_ngram_overall_overallCAL.Rda") #identical to contextualized n-gram overall# 
saveRDS (RF_fit_fulldtm_overall_overallCA, "RF_fit_fulldtm_overall_overallCA.Rda")

cal_words_fulldtm_overall_CAL <- readRDS("cal_words_ngrams_CAL.Rda") #identical to contextualized n-gram overall# 
cal_words_fulldtm_overall_CAL <- dplyr::filter (cal_words_fulldtm_overall_CAL, Dimension=='overall')
colnames(cal_words_fulldtm_overall_CAL)[1] <- 'cor_df[, 3]'
nrow (cal_words_fulldtm_overall_CAL)

###
### Making Word List for all Great 8 Dimenions ###
###
cal_words_fulldtm_G1_CAL$Dimension <- 'G1'
cal_words_fulldtm_G2_CAL$Dimension <- 'G2'
cal_words_fulldtm_G3_CAL$Dimension <- 'G3'
cal_words_fulldtm_G4_CAL$Dimension <- 'G4'
cal_words_fulldtm_G5_CAL$Dimension <- 'G5'
cal_words_fulldtm_G6_CAL$Dimension <- 'G6'
cal_words_fulldtm_G7_CAL$Dimension <- 'G7'
cal_words_fulldtm_G8_CAL$Dimension <- 'G8'
cal_words_fulldtm_overall_CAL$Dimension <- 'overall'
cal_words_fulldtm_CAL <- rbind (cal_words_fulldtm_G1_CAL, cal_words_fulldtm_G2_CAL, cal_words_fulldtm_G3_CAL, cal_words_fulldtm_G4_CAL, cal_words_fulldtm_G5_CAL, 
                                cal_words_fulldtm_G6_CAL, cal_words_fulldtm_G7_CAL, cal_words_fulldtm_G8_CAL, cal_words_fulldtm_overall_CAL)
colnames(cal_words_fulldtm_CAL)[1] <- 'word'
nrow (cal_words_fulldtm_CAL)
saveRDS(cal_words_fulldtm_CAL, "cal_words_fulldtm_CAL.Rda")

############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
#################################################   Building Models based on Word Embeddings  ###################################################
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################

######
###### G1    #####
#######
df <- readRDS ("Embed_df.Rda")
modeling_df <- dplyr::select (df, Ratee_G1_num, Word_count:Glove_300)
modeling_df <- as.data.frame (apply (modeling_df, 2, scale)) #normalizing
dim (modeling_df)

num_cores <- detectCores() - 1 # Run in parallel across processors on local machine
cl.spec <- rep("localhost", num_cores) # Run in parallel across processors on local machine
cl <- parallel::makeCluster(cl.spec) # Run in parallel across processors on local machine
registerDoParallel(cl, cores=3)
set.seed(333333)
#fit  <- tuneRF(modeling_df[,2:ncol(modeling_df)], y=modeling_df[,1], mtryStart=(ncol(modeling_df)/3),ntreeTry=2000,stepFactor=1.25,improve=.00001) 
#fit
rf_model <-  randomForest (Ratee_G1_num ~ ., data = modeling_df, nodesize=5, ntree=2000, mtry= 125)
rf_model
saveRDS(rf_model, "RF_fit_glove_G1_CAL.Rda")
stopCluster(cl)
closeAllConnections()
registerDoSEQ()

######
###### G2    #####
#######
df <- readRDS ("Embed_df.Rda")
modeling_df <- dplyr::select (df, Ratee_G2_num, Word_count:Glove_300)
modeling_df <- as.data.frame (apply (modeling_df, 2, scale)) #normalizing
dim (modeling_df)

num_cores <- detectCores() - 1 # Run in parallel across processors on local machine
cl.spec <- rep("localhost", num_cores) # Run in parallel across processors on local machine
cl <- parallel::makeCluster(cl.spec) # Run in parallel across processors on local machine
registerDoParallel(cl, cores=3)
set.seed(333333)
#fit  <- tuneRF(modeling_df[,2:ncol(modeling_df)], y=modeling_df[,1], mtryStart=(ncol(modeling_df)/3),ntreeTry=2000,stepFactor=1.25,improve=.00001) 
#fit
rf_model <-  randomForest (Ratee_G2_num ~ ., data = modeling_df, nodesize=5, ntree=2000, mtry= 156)
rf_model
saveRDS(rf_model, "RF_fit_glove_G2_CAL.Rda")
stopCluster(cl)
closeAllConnections()
registerDoSEQ()

######
###### G3    #####
#######
df <- readRDS ("Embed_df.Rda")
modeling_df <- dplyr::select (df, Ratee_G3_num, Word_count:Glove_300)
modeling_df <- as.data.frame (apply (modeling_df, 2, scale)) #normalizing
dim (modeling_df)

num_cores <- detectCores() - 1 # Run in parallel across processors on local machine
cl.spec <- rep("localhost", num_cores) # Run in parallel across processors on local machine
cl <- parallel::makeCluster(cl.spec) # Run in parallel across processors on local machine
registerDoParallel(cl, cores=3)
set.seed(333333)
#fit  <- tuneRF(modeling_df[,2:ncol(modeling_df)], y=modeling_df[,1], mtryStart=(ncol(modeling_df)/3),ntreeTry=2000,stepFactor=1.25,improve=.00001) 
#fit
rf_model <-  randomForest (Ratee_G3_num ~ ., data = modeling_df, nodesize=5, ntree=2000, mtry= 81)
rf_model
saveRDS(rf_model, "RF_fit_glove_G3_CAL.Rda")
stopCluster(cl)
closeAllConnections()
registerDoSEQ()

######
###### G4    #####
#######
df <- readRDS ("Embed_df.Rda")
modeling_df <- dplyr::select (df, Ratee_G4_num, Word_count:Glove_300)
modeling_df <- as.data.frame (apply (modeling_df, 2, scale)) #normalizing
dim (modeling_df)

num_cores <- detectCores() - 1 # Run in parallel across processors on local machine
cl.spec <- rep("localhost", num_cores) # Run in parallel across processors on local machine
cl <- parallel::makeCluster(cl.spec) # Run in parallel across processors on local machine
registerDoParallel(cl, cores=3)
set.seed(333333)
#fit  <- tuneRF(modeling_df[,2:ncol(modeling_df)], y=modeling_df[,1], mtryStart=(ncol(modeling_df)/3),ntreeTry=2000,stepFactor=1.25,improve=.00001) 
#fit
rf_model <-  randomForest (Ratee_G4_num ~ ., data = modeling_df, nodesize=5, ntree=2000, mtry= 101)
rf_model
saveRDS(rf_model, "RF_fit_glove_G4_CAL.Rda")
stopCluster(cl)
closeAllConnections()
registerDoSEQ()

######
###### G5    #####
#######
df <- readRDS ("Embed_df.Rda")
modeling_df <- dplyr::select (df, Ratee_G5_num, Word_count:Glove_300)
modeling_df <- as.data.frame (apply (modeling_df, 2, scale)) #normalizing
dim (modeling_df)

num_cores <- detectCores() - 1 # Run in parallel across processors on local machine
cl.spec <- rep("localhost", num_cores) # Run in parallel across processors on local machine
cl <- parallel::makeCluster(cl.spec) # Run in parallel across processors on local machine
registerDoParallel(cl, cores=3)
set.seed(333333)
#fit  <- tuneRF(modeling_df[,2:ncol(modeling_df)], y=modeling_df[,1], mtryStart=(ncol(modeling_df)/3),ntreeTry=2000,stepFactor=1.25,improve=.00001) 
#fit
rf_model <-  randomForest (Ratee_G5_num ~ ., data = modeling_df, nodesize=5, ntree=2000, mtry= 101)
rf_model
saveRDS(rf_model, "RF_fit_glove_G5_CAL.Rda")
stopCluster(cl)
closeAllConnections()
registerDoSEQ()


######
###### G6    #####
#######
df <- readRDS ("Embed_df.Rda")
modeling_df <- dplyr::select (df, Ratee_G6_num, Word_count:Glove_300)
modeling_df <- as.data.frame (apply (modeling_df, 2, scale)) #normalizing
dim (modeling_df)

num_cores <- detectCores() - 1 # Run in parallel across processors on local machine
cl.spec <- rep("localhost", num_cores) # Run in parallel across processors on local machine
cl <- parallel::makeCluster(cl.spec) # Run in parallel across processors on local machine
registerDoParallel(cl, cores=3)
set.seed(333333)
#fit  <- tuneRF(modeling_df[,2:ncol(modeling_df)], y=modeling_df[,1], mtryStart=(ncol(modeling_df)/3),ntreeTry=2000,stepFactor=1.25,improve=.00001) 
#fit
rf_model <-  randomForest (Ratee_G6_num ~ ., data = modeling_df, nodesize=5, ntree=2000, mtry= 125)
rf_model
saveRDS(rf_model, "RF_fit_glove_G6_CAL.Rda")
stopCluster(cl)
closeAllConnections()
registerDoSEQ()

######
###### G7    #####
#######
df <- readRDS ("Embed_df.Rda")
modeling_df <- dplyr::select (df, Ratee_G7_num, Word_count:Glove_300)
modeling_df <- as.data.frame (apply (modeling_df, 2, scale)) #normalizing
dim (modeling_df)

num_cores <- detectCores() - 1 # Run in parallel across processors on local machine
cl.spec <- rep("localhost", num_cores) # Run in parallel across processors on local machine
cl <- parallel::makeCluster(cl.spec) # Run in parallel across processors on local machine
registerDoParallel(cl, cores=3)
set.seed(333333)
#fit  <- tuneRF(modeling_df[,2:ncol(modeling_df)], y=modeling_df[,1], mtryStart=(ncol(modeling_df)/3),ntreeTry=2000,stepFactor=1.25,improve=.00001) 
#fit
rf_model <-  randomForest (Ratee_G7_num ~ ., data = modeling_df, nodesize=5, ntree=2000, mtry= 101)
rf_model
saveRDS(rf_model, "RF_fit_glove_G7_CAL.Rda")
stopCluster(cl)
closeAllConnections()
registerDoSEQ()

######
###### G8    #####
#######
df <- readRDS ("Embed_df.Rda")
modeling_df <- dplyr::select (df, Ratee_G8_num, Word_count:Glove_300)
modeling_df <- as.data.frame (apply (modeling_df, 2, scale)) #normalizing
dim (modeling_df)

num_cores <- detectCores() - 1 # Run in parallel across processors on local machine
cl.spec <- rep("localhost", num_cores) # Run in parallel across processors on local machine
cl <- parallel::makeCluster(cl.spec) # Run in parallel across processors on local machine
registerDoParallel(cl, cores=3)
set.seed(333333)
#fit  <- tuneRF(modeling_df[,2:ncol(modeling_df)], y=modeling_df[,1], mtryStart=(ncol(modeling_df)/3),ntreeTry=2000,stepFactor=1.25,improve=.00001) 
#fit
rf_model <-  randomForest (Ratee_G8_num ~ ., data = modeling_df, nodesize=5, ntree=2000, mtry= 101)
rf_model
saveRDS(rf_model, "RF_fit_glove_G8_CAL.Rda")
stopCluster(cl)
closeAllConnections()
registerDoSEQ()

######
###### overall    #####
#######
df <- readRDS ("Embed_df.Rda")
modeling_df <- dplyr::select (df, Performance_Composite, Word_count:Glove_300)
modeling_df <- as.data.frame (apply (modeling_df, 2, scale)) #normalizing
dim (modeling_df)

num_cores <- detectCores() - 1 # Run in parallel across processors on local machine
cl.spec <- rep("localhost", num_cores) # Run in parallel across processors on local machine
cl <- parallel::makeCluster(cl.spec) # Run in parallel across processors on local machine
registerDoParallel(cl, cores=3)
set.seed(333333)
#fit  <- tuneRF(modeling_df[,2:ncol(modeling_df)], y=modeling_df[,1], mtryStart=(ncol(modeling_df)/3),ntreeTry=2000,stepFactor=1.25,improve=.00001) 
#fit
rf_model <-  randomForest (Performance_Composite ~ ., data = modeling_df, nodesize=5, ntree=2000, mtry= 125)
rf_model
saveRDS(rf_model, "RF_fit_glove_overall_CAL.Rda")
stopCluster(cl)
closeAllConnections()
registerDoSEQ()

# You have now created new aglorithms across the Great 8 and various NLP methods #





