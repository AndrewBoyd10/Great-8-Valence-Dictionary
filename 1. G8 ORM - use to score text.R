### This syntax will start-to-finish take text data and produce theme and valence scores based on Speer (2020) ###
### The output is your original input file + newly created theme and valence scores for all the Great 8 and overall job performance ###

### All these files can be found on the following GitHub: https://github.com/AndrewBoyd10/Great-8-Valence-Dictionary or OSF (https://osf.io/2wcsn/)
### To begin, copy over the following files from https://github.com/AndrewBoyd10/Great-8-Valence-Dictionary  into a folder on your local computer. This folder will be your working directory:
    #"cal_words" files, which contain word phrases used in random forests models
    # "Great 8 Narrative Dictionary - use": lemmatized theme words used to perform n-gram scoring and to create theme scores
    # "Norm" files: used to adjust scores as last step
    # 27 "RF" objects which contain the algorithms to score data into valence scores
    # Probably easiest to just copy all files into your local working directory

### You will also need to create an R dataframe, saved in your local directory, and named "Text_df" that contains the following variables:
    #Narrative = text (such that "Narrative" is the variable name)
    #Unique_ID_Ratee = ID for a given person rated
    #Ratee_G1_num thru Ratee_G8_num = optional variables reflecting numerical performance ratings for Great 8 dimensions 1-8. Only needed if training, though correlations with these are calculated in this syntax
    #Performance_Composite = optional variable reflecting performance composite of all numerical performance ratings. Only needed if training, though correlations with this are calculated in this syntax

### Finally, you'll need to download the "glove.42B.300d" file from https://nlp.stanford.edu/projects/glove/ and save the file as "glove.42B.300d.txt" in your local folder ###

############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
#################################################   Preprocessing Text  #####################################################################
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################

# load packages and set working directory to your folder location #
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

library(randomForest)

setwd("INSERT_FOLDER_LOCATION") #setting working directory#

### Dictionary ###
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

##### Storing Text Object for Word Embedding Analysis  ######
Text_df_Embed <- Text_df

### More Cleaning (more will be done later below) ###
Text_df$Narrative <- gsub("\\!"," exclamation_mark ", Text_df$Narrative)
Text_df$Narrative <- gsub("!"," exclamation_mark ", Text_df$Narrative)
Text_df$Narrative <- gsub("-"," ", Text_df$Narrative) #helps with theme match
Text_df$Narrative <- gsub("\\?"," question_mark ", Text_df$Narrative)

#word and character count#
Text_df$Word_count <- str_count(Text_df$Narrative, "\\w+")
Text_df$Characters <- nchar (Text_df$Narrative)

###
#Sentence Breakdown (automatically lower-cases text) & Additional Cleaning #
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
sentences$sentence <- lemmatize_strings(sentences$sentence)
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
#################################################  Creating SCores: Contextualized N-Gram and Theme Scores  ###################################################
############################################################################################################################################
############################################################################################################################################

#We've already loaded the Great 8 dictionaries, which will now be applied here. Below, searches are made for dictionary-specific phrases and then those sentences are used

###
### G1 Score ###
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
colnames(G1_file)
G1_wordCorpus <- VCorpus(DataframeSource(G1_file))
tokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 1, max = 2)) #bigrams as max
dtm <- DocumentTermMatrix (G1_wordCorpus, control = list (tokenize=tokenizer))
dtm <- weightBin(dtm) #dichotomous weighting
dtm_m <- as.matrix(dtm)

#Filtering dtm 
cor_df <- dplyr::filter (readRDS("cal_words_ngrams_CAL.Rda"), Dimension == 'G1') #filtering to set of words identified in model training 
dim (cor_df)
dtm_m <- dtm_m [,colnames(dtm_m) %in% cor_df[,1]]
ncol(dtm_m)

#creating empty variables for terms that did not appear in your new local file #
add_ons <- list()
for (word in cor_df [,1]) {
  ifelse (word %in% colnames(dtm_m), NA, print(word))
  ifelse (word %in% colnames(dtm_m), NA, add_ons[[word]] <- word)
}
Blank_Columns  <-   data.frame(matrix(0, nrow = nrow(dtm_m), ncol = length(add_ons)))
colnames (Blank_Columns) <- add_ons
dtm_m <- cbind (dtm_m, Blank_Columns)
dtm_m <- dtm_m[,order(colnames(dtm_m))] #ordering so same for algorithms
colnames(dtm_m)

Word_count <- str_count(G1_file$text, "\\w+")
Word_count <- as.data.frame(Word_count)
names (Word_count) [1] <- 'Word_Count_G1'
dtm_m <- cbind (Word_count, dtm_m)
dtm_m [1:10, 1:10]
dim (dtm_m)

#Scoring: Random Forests for N-gram#
model <- readRDS("RF_fit_ngram_G1_overallCAL.Rda") #loading in previously trained model#
modeling_df <- dtm_m
colnames(modeling_df) <- paste(colnames(modeling_df), "_rf", sep = "")
colnames(modeling_df) <- gsub(" ", "__", colnames(modeling_df))
modeling_df <- as.data.frame (apply (modeling_df, 2, scale)) #normalizing#
modeling_df [is.na(modeling_df)] <- 0 #have to for standardizing if there is no variance in new local file for phrase
modeling_df [1:10, 1:10]
Predicted_G1 <- as.data.frame (predict(model, newdata = modeling_df))
colnames(Predicted_G1) [1] <- 'RF_ngram_pred_G1'
Text_df <- cbind (Text_df, Predicted_G1)
psych::corr.test (Text_df[,c("Ratee_G1_num",  "RF_ngram_pred_G1")])

###
### G2 Score ###
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
colnames(G2_file)
G2_wordCorpus <- VCorpus(DataframeSource(G2_file))
tokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 1, max = 2)) #bigrams as max
dtm <- DocumentTermMatrix (G2_wordCorpus, control = list (tokenize=tokenizer))
dtm <- weightBin(dtm) #dichotomous weighting
dtm_m <- as.matrix(dtm)

#Filtering dtm 
cor_df <- dplyr::filter (readRDS("cal_words_ngrams_CAL.Rda"), Dimension == 'G2')
dim (cor_df)
dtm_m <- dtm_m [,colnames(dtm_m) %in% cor_df[,1]]
ncol(dtm_m)

#creating empty variables for terms that did not appear in your new local file #
add_ons <- list()
for (word in cor_df [,1]) {
  ifelse (word %in% colnames(dtm_m), NA, print(word))
  ifelse (word %in% colnames(dtm_m), NA, add_ons[[word]] <- word)
}
Blank_Columns  <-   data.frame(matrix(0, nrow = nrow(dtm_m), ncol = length(add_ons)))
colnames (Blank_Columns) <- add_ons
dtm_m <- cbind (dtm_m, Blank_Columns)
dtm_m <- dtm_m[,order(colnames(dtm_m))] #ordering so same for algorithms
colnames(dtm_m)

Word_count <- str_count(G2_file$text, "\\w+")
Word_count <- as.data.frame(Word_count)
names (Word_count) [1] <- 'Word_Count_G2'
dtm_m <- cbind (Word_count, dtm_m)
dtm_m [1:10, 1:10]
dim (dtm_m)

#Scoring: Random Forests for N-gram#
model <- readRDS("RF_fit_ngram_G2_overallCAL.Rda")
modeling_df <- dtm_m
colnames(modeling_df) <- paste(colnames(modeling_df), "_rf", sep = "")
colnames(modeling_df) <- gsub(" ", "__", colnames(modeling_df))
modeling_df <- as.data.frame (apply (modeling_df, 2, scale)) #normalizing#
modeling_df [is.na(modeling_df)] <- 0 #have to for standardizing
modeling_df [1:10, 1:10]
Predicted_G2 <- as.data.frame (predict(model, newdata = modeling_df))
colnames(Predicted_G2) [1] <- 'RF_ngram_pred_G2'
Text_df <- cbind (Text_df, Predicted_G2)
psych::corr.test (Text_df[,c("Ratee_G2_num",  "RF_ngram_pred_G2")])

###
### G3 Score ###
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
colnames(G3_file)
G3_wordCorpus <- VCorpus(DataframeSource(G3_file))
tokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 1, max = 2)) #bigrams as max
dtm <- DocumentTermMatrix (G3_wordCorpus, control = list (tokenize=tokenizer))
dtm <- weightBin(dtm) #dichotomous weighting
dtm_m <- as.matrix(dtm)

#Filtering dtm 
cor_df <- dplyr::filter (readRDS("cal_words_ngrams_CAL.Rda"), Dimension == 'G3')
dim (cor_df)
dtm_m <- dtm_m [,colnames(dtm_m) %in% cor_df[,1]]
ncol(dtm_m)

#creating empty variables for terms that did not appear in your new local file #
add_ons <- list()
for (word in cor_df [,1]) {
  ifelse (word %in% colnames(dtm_m), NA, print(word))
  ifelse (word %in% colnames(dtm_m), NA, add_ons[[word]] <- word)
}
Blank_Columns  <-   data.frame(matrix(0, nrow = nrow(dtm_m), ncol = length(add_ons)))
colnames (Blank_Columns) <- add_ons
dtm_m <- cbind (dtm_m, Blank_Columns)
dtm_m <- dtm_m[,order(colnames(dtm_m))] #ordering so same for algorithms
colnames(dtm_m)

Word_count <- str_count(G3_file$text, "\\w+")
Word_count <- as.data.frame(Word_count)
names (Word_count) [1] <- 'Word_Count_G3'
dtm_m <- cbind (Word_count, dtm_m)
dtm_m [1:10, 1:10]
dim (dtm_m)

#Scoring: Random Forests for N-gram#
model <- readRDS("RF_fit_ngram_G3_overallCAL.Rda")
modeling_df <- dtm_m
colnames(modeling_df) <- paste(colnames(modeling_df), "_rf", sep = "")
colnames(modeling_df) <- gsub(" ", "__", colnames(modeling_df))
modeling_df <- as.data.frame (apply (modeling_df, 2, scale)) #normalizing#
modeling_df [is.na(modeling_df)] <- 0 #have to for standardizing
modeling_df [1:10, 1:10]
Predicted_G3 <- as.data.frame (predict(model, newdata = modeling_df))
colnames(Predicted_G3) [1] <- 'RF_ngram_pred_G3'
Text_df <- cbind (Text_df, Predicted_G3)
psych::corr.test (Text_df[,c("Ratee_G3_num",  "RF_ngram_pred_G3")])

###
### G4 Score ###
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
colnames(G4_file)
G4_wordCorpus <- VCorpus(DataframeSource(G4_file))
tokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 1, max = 2)) #bigrams as max
dtm <- DocumentTermMatrix (G4_wordCorpus, control = list (tokenize=tokenizer))
dtm <- weightBin(dtm) #dichotomous weighting
dtm_m <- as.matrix(dtm)

#Filtering dtm 
cor_df <- dplyr::filter (readRDS("cal_words_ngrams_CAL.Rda"), Dimension == 'G4')
dim (cor_df)
dtm_m <- dtm_m [,colnames(dtm_m) %in% cor_df[,1]]
ncol(dtm_m)

#creating empty variables for terms that did not appear in your new local file #
add_ons <- list()
for (word in cor_df [,1]) {
  ifelse (word %in% colnames(dtm_m), NA, print(word))
  ifelse (word %in% colnames(dtm_m), NA, add_ons[[word]] <- word)
}
Blank_Columns  <-   data.frame(matrix(0, nrow = nrow(dtm_m), ncol = length(add_ons)))
colnames (Blank_Columns) <- add_ons
dtm_m <- cbind (dtm_m, Blank_Columns)
dtm_m <- dtm_m[,order(colnames(dtm_m))] #ordering so same for algorithms
colnames(dtm_m)

Word_count <- str_count(G4_file$text, "\\w+")
Word_count <- as.data.frame(Word_count)
names (Word_count) [1] <- 'Word_Count_G4'
dtm_m <- cbind (Word_count, dtm_m)
dtm_m [1:10, 1:10]
dim (dtm_m)

#Scoring: Random Forests for N-gram#
model <- readRDS("RF_fit_ngram_G4_overallCAL.Rda")
modeling_df <- dtm_m
colnames(modeling_df) <- paste(colnames(modeling_df), "_rf", sep = "")
colnames(modeling_df) <- gsub(" ", "__", colnames(modeling_df))
modeling_df <- as.data.frame (apply (modeling_df, 2, scale)) #normalizing#
modeling_df [is.na(modeling_df)] <- 0 #have to for standardizing
modeling_df [1:10, 1:10]
Predicted_G4 <- as.data.frame (predict(model, newdata = modeling_df))
colnames(Predicted_G4) [1] <- 'RF_ngram_pred_G4'
Text_df <- cbind (Text_df, Predicted_G4)
psych::corr.test (Text_df[,c("Ratee_G4_num",  "RF_ngram_pred_G4")])

###
### G5 Score ###
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
colnames(G5_file)
G5_wordCorpus <- VCorpus(DataframeSource(G5_file))
tokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 1, max = 2)) #bigrams as max
dtm <- DocumentTermMatrix (G5_wordCorpus, control = list (tokenize=tokenizer))
dtm <- weightBin(dtm) #dichotomous weighting
dtm_m <- as.matrix(dtm)

#Filtering dtm 
cor_df <- dplyr::filter (readRDS("cal_words_ngrams_CAL.Rda"), Dimension == 'G5')
dim (cor_df)
dtm_m <- dtm_m [,colnames(dtm_m) %in% cor_df[,1]]
ncol(dtm_m)

#creating empty variables for terms that did not appear in your new local file #
add_ons <- list()
for (word in cor_df [,1]) {
  ifelse (word %in% colnames(dtm_m), NA, print(word))
  ifelse (word %in% colnames(dtm_m), NA, add_ons[[word]] <- word)
}
Blank_Columns  <-   data.frame(matrix(0, nrow = nrow(dtm_m), ncol = length(add_ons)))
colnames (Blank_Columns) <- add_ons
dtm_m <- cbind (dtm_m, Blank_Columns)
dtm_m <- dtm_m[,order(colnames(dtm_m))] #ordering so same for algorithms
colnames(dtm_m)

Word_count <- str_count(G5_file$text, "\\w+")
Word_count <- as.data.frame(Word_count)
names (Word_count) [1] <- 'Word_Count_G5'
dtm_m <- cbind (Word_count, dtm_m)
dtm_m [1:10, 1:10]
dim (dtm_m)

#Scoring: Random Forests for N-gram#
model <- readRDS("RF_fit_ngram_G5_overallCAL.Rda")
modeling_df <- dtm_m
colnames(modeling_df) <- paste(colnames(modeling_df), "_rf", sep = "")
colnames(modeling_df) <- gsub(" ", "__", colnames(modeling_df))
modeling_df <- as.data.frame (apply (modeling_df, 2, scale)) #normalizing#
modeling_df [is.na(modeling_df)] <- 0 #have to for standardizing
modeling_df [1:10, 1:10]
Predicted_G5 <- as.data.frame (predict(model, newdata = modeling_df))
colnames(Predicted_G5) [1] <- 'RF_ngram_pred_G5'
Text_df <- cbind (Text_df, Predicted_G5)
psych::corr.test (Text_df[,c("Ratee_G5_num",  "RF_ngram_pred_G5")])

###
### G6 Score ###
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
colnames(G6_file)
G6_wordCorpus <- VCorpus(DataframeSource(G6_file))
tokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 1, max = 2)) #bigrams as max
dtm <- DocumentTermMatrix (G6_wordCorpus, control = list (tokenize=tokenizer))
dtm <- weightBin(dtm) #dichotomous weighting
dtm_m <- as.matrix(dtm)

#Filtering dtm 
cor_df <- dplyr::filter (readRDS("cal_words_ngrams_CAL.Rda"), Dimension == 'G6')
dim (cor_df)
dtm_m <- dtm_m [,colnames(dtm_m) %in% cor_df[,1]]
ncol(dtm_m)

#creating empty variables for terms that did not appear in your new local file #
add_ons <- list()
for (word in cor_df [,1]) {
  ifelse (word %in% colnames(dtm_m), NA, print(word))
  ifelse (word %in% colnames(dtm_m), NA, add_ons[[word]] <- word)
}
Blank_Columns  <-   data.frame(matrix(0, nrow = nrow(dtm_m), ncol = length(add_ons)))
colnames (Blank_Columns) <- add_ons
dtm_m <- cbind (dtm_m, Blank_Columns)
dtm_m <- dtm_m[,order(colnames(dtm_m))] #ordering so same for algorithms
colnames(dtm_m)

Word_count <- str_count(G6_file$text, "\\w+")
Word_count <- as.data.frame(Word_count)
names (Word_count) [1] <- 'Word_Count_G6'
dtm_m <- cbind (Word_count, dtm_m)
dtm_m [1:10, 1:10]
dim (dtm_m)

#Scoring: Random Forests for N-gram#
model <- readRDS("RF_fit_ngram_G6_overallCAL.Rda")
modeling_df <- dtm_m
colnames(modeling_df) <- paste(colnames(modeling_df), "_rf", sep = "")
colnames(modeling_df) <- gsub(" ", "__", colnames(modeling_df))
modeling_df <- as.data.frame (apply (modeling_df, 2, scale)) #normalizing#
modeling_df [is.na(modeling_df)] <- 0 #have to for standardizing
modeling_df [1:10, 1:10]
Predicted_G6 <- as.data.frame (predict(model, newdata = modeling_df))
colnames(Predicted_G6) [1] <- 'RF_ngram_pred_G6'
Text_df <- cbind (Text_df, Predicted_G6)
psych::corr.test (Text_df[,c("Ratee_G6_num",  "RF_ngram_pred_G6")])

###
### G7 Score ###
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
colnames(G7_file)
G7_wordCorpus <- VCorpus(DataframeSource(G7_file))
tokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 1, max = 2)) #bigrams as max
dtm <- DocumentTermMatrix (G7_wordCorpus, control = list (tokenize=tokenizer))
dtm <- weightBin(dtm) #dichotomous weighting
dtm_m <- as.matrix(dtm)

#Filtering dtm 
cor_df <- dplyr::filter (readRDS("cal_words_ngrams_CAL.Rda"), Dimension == 'G7')
dim (cor_df)
dtm_m <- dtm_m [,colnames(dtm_m) %in% cor_df[,1]]
ncol(dtm_m)

#creating empty variables for terms that did not appear in your new local file #
add_ons <- list()
for (word in cor_df [,1]) {
  ifelse (word %in% colnames(dtm_m), NA, print(word))
  ifelse (word %in% colnames(dtm_m), NA, add_ons[[word]] <- word)
}
Blank_Columns  <-   data.frame(matrix(0, nrow = nrow(dtm_m), ncol = length(add_ons)))
colnames (Blank_Columns) <- add_ons
dtm_m <- cbind (dtm_m, Blank_Columns)
dtm_m <- dtm_m[,order(colnames(dtm_m))] #ordering so same for algorithms
colnames(dtm_m)

Word_count <- str_count(G7_file$text, "\\w+")
Word_count <- as.data.frame(Word_count)
names (Word_count) [1] <- 'Word_Count_G7'
dtm_m <- cbind (Word_count, dtm_m)
dtm_m [1:10, 1:10]
dim (dtm_m)

#Scoring: Random Forests for N-gram#
model <- readRDS("RF_fit_ngram_G7_overallCAL.Rda")
modeling_df <- dtm_m
colnames(modeling_df) <- paste(colnames(modeling_df), "_rf", sep = "")
colnames(modeling_df) <- gsub(" ", "__", colnames(modeling_df))
modeling_df <- as.data.frame (apply (modeling_df, 2, scale)) #normalizing#
modeling_df [is.na(modeling_df)] <- 0 #have to for standardizing
modeling_df [1:10, 1:10]
Predicted_G7 <- as.data.frame (predict(model, newdata = modeling_df))
colnames(Predicted_G7) [1] <- 'RF_ngram_pred_G7'
Text_df <- cbind (Text_df, Predicted_G7)
psych::corr.test (Text_df[,c("Ratee_G7_num",  "RF_ngram_pred_G7")])

###
### G8 Score ###
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
colnames(G8_file)
G8_wordCorpus <- VCorpus(DataframeSource(G8_file))
tokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 1, max = 2)) #bigrams as max
dtm <- DocumentTermMatrix (G8_wordCorpus, control = list (tokenize=tokenizer))
dtm <- weightBin(dtm) #dichotomous weighting
dtm_m <- as.matrix(dtm)

#Filtering dtm 
cor_df <- dplyr::filter (readRDS("cal_words_ngrams_CAL.Rda"), Dimension == 'G8')
dim (cor_df)
dtm_m <- dtm_m [,colnames(dtm_m) %in% cor_df[,1]]
ncol(dtm_m)

#creating empty variables for terms that did not appear in your new local file #
add_ons <- list()
for (word in cor_df [,1]) {
  ifelse (word %in% colnames(dtm_m), NA, print(word))
  ifelse (word %in% colnames(dtm_m), NA, add_ons[[word]] <- word)
}
Blank_Columns  <-   data.frame(matrix(0, nrow = nrow(dtm_m), ncol = length(add_ons)))
colnames (Blank_Columns) <- add_ons
dtm_m <- cbind (dtm_m, Blank_Columns)
dtm_m <- dtm_m[,order(colnames(dtm_m))] #ordering so same for algorithms
colnames(dtm_m)

Word_count <- str_count(G8_file$text, "\\w+")
Word_count <- as.data.frame(Word_count)
names (Word_count) [1] <- 'Word_Count_G8'
dtm_m <- cbind (Word_count, dtm_m)
dtm_m [1:10, 1:10]
dim (dtm_m)

#Scoring: Random Forests for N-gram#
model <- readRDS("RF_fit_ngram_G8_overallCAL.Rda")
modeling_df <- dtm_m
colnames(modeling_df) <- paste(colnames(modeling_df), "_rf", sep = "")
colnames(modeling_df) <- gsub(" ", "__", colnames(modeling_df))
modeling_df <- as.data.frame (apply (modeling_df, 2, scale)) #normalizing#
modeling_df [is.na(modeling_df)] <- 0 #have to for standardizing
modeling_df [1:10, 1:10]
Predicted_G8 <- as.data.frame (predict(model, newdata = modeling_df))
colnames(Predicted_G8) [1] <- 'RF_ngram_pred_G8'
Text_df <- cbind (Text_df, Predicted_G8)
psych::corr.test (Text_df[,c("Ratee_G8_num",  "RF_ngram_pred_G8")])

###
### Overall Composite Score ###
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
dtm <- weightBin(dtm) #dichotomous weighting
dtm_m <- as.matrix(dtm)

#Filtering dtm 
cor_df <- dplyr::filter (readRDS("cal_words_ngrams_CAL.Rda"), Dimension == 'overall')
dim (cor_df)
dtm_m <- dtm_m [,colnames(dtm_m) %in% cor_df[,1]]
ncol(dtm_m)

#creating empty variables for terms that did not appear in your new local file #
add_ons <- list()
for (word in cor_df [,1]) {
  ifelse (word %in% colnames(dtm_m), NA, print(word))
  ifelse (word %in% colnames(dtm_m), NA, add_ons[[word]] <- word)
}
Blank_Columns  <-   data.frame(matrix(0, nrow = nrow(dtm_m), ncol = length(add_ons)))
colnames (Blank_Columns) <- add_ons
dtm_m <- cbind (dtm_m, Blank_Columns)
dtm_m <- dtm_m[,order(colnames(dtm_m))] #ordering so same for algorithms
colnames(dtm_m)

Word_count <- str_count(Overall_file$text, "\\w+")
Word_count <- as.data.frame(Word_count)
names (Word_count) [1] <- 'Word_Count_overall'
dtm_m <- cbind (Word_count, dtm_m)
dtm_m [1:10, 1:10]
dim (dtm_m)

#Scoring: Random Forests for N-gram#
model <- readRDS("RF_fit_ngram_overall_overallCAL.Rda")
modeling_df <- dtm_m
colnames(modeling_df) <- paste(colnames(modeling_df), "_rf", sep = "")
colnames(modeling_df) <- gsub(" ", "__", colnames(modeling_df))
modeling_df <- as.data.frame (apply (modeling_df, 2, scale)) #normalizing#
modeling_df [is.na(modeling_df)] <- 0 #have to for standardizing
modeling_df [1:10, 1:10]
Predicted_overall <- as.data.frame (predict(model, newdata = modeling_df))
colnames(Predicted_overall) [1] <- 'RF_ngram_pred_overall'
Text_df <- cbind (Text_df, Predicted_overall)
psych::corr.test (Text_df[,c("Performance_Composite",  "RF_ngram_pred_overall")])

###
### Combining Scores: Ngram ###
###
Ngram_Scores <- cbind (Predicted_G1, Predicted_G2, Predicted_G3, Predicted_G4, Predicted_G5, Predicted_G6, Predicted_G7, Predicted_G8, Predicted_overall)
colnames(Ngram_Scores)
psych::corr.test(Ngram_Scores)
dim(Ngram_Scores)
saveRDS(Ngram_Scores, "Ngram_Scores.Rda")

###
### Saving Theme Scores ###
###
Theme <- dplyr::select (Text_df, Word_count:Ratee_total_sentences, G1_Theme_Score, G2_Theme_Score, G3_Theme_Score, G4_Theme_Score, G5_Theme_Score, 
                        G6_Theme_Score, G7_Theme_Score, G8_Theme_Score)
saveRDS(Theme, "Theme_Scores.Rda")

############################################################################################################################################
############################################################################################################################################
#################################################  Creating SCores: Traditional BOW (Full DTM)  ###################################################
############################################################################################################################################
############################################################################################################################################

Overall_file <- sentences
Overall_file <- aggregate (Overall_file [,2], by=list (Overall_file$Unique_ID_Ratee), paste, collapse = " ")
names(Overall_file) <- c('doc_id', 'text')
Overall_file$text <- stripWhitespace (Overall_file$text)

Overall_wordCorpus <- VCorpus(DataframeSource(Overall_file))
tokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 1, max = 2))
dtm <- DocumentTermMatrix (Overall_wordCorpus, control = list (tokenize=tokenizer))
dtm <- weightBin(dtm) #dichotomous weighting
dtm_m <- as.matrix(dtm)
dtm_m <- cbind (dplyr::select(Text_df, Word_count), dtm_m)
dtm_m_base <- dtm_m #storing this to re-use later

###
### G1 Score ###
###

#Filtering dtm 
dtm_m <- dtm_m_base
cor_df <- dplyr::filter (readRDS("cal_words_fulldtm_CAL.Rda"), Dimension == 'G1')
dim (cor_df)
dtm_m <- dtm_m [,colnames(dtm_m) %in% cor_df[,1]]
ncol(dtm_m)

#creating empty variables for terms that did not appear in your new local file #
add_ons <- list()
for (word in cor_df [,1]) {
  ifelse (word %in% colnames(dtm_m), NA, print(word))
  ifelse (word %in% colnames(dtm_m), NA, add_ons[[word]] <- word)
}
Blank_Columns  <-   data.frame(matrix(0, nrow = nrow(dtm_m), ncol = length(add_ons)))
colnames (Blank_Columns) <- add_ons
dtm_m <- cbind (dtm_m, Blank_Columns)
dtm_m <- dtm_m[,order(colnames(dtm_m))] #ordering so same for algorithms

#creating scores#
model <- readRDS("RF_fit_fulldtm_G1_overallCAL.Rda")
modeling_df <- dtm_m
colnames(modeling_df) <- paste(colnames(modeling_df), "_rf", sep = "")
colnames(modeling_df) <- gsub(" ", "__", colnames(modeling_df))
modeling_df <- as.data.frame (apply (modeling_df, 2, scale)) #normalizing#
modeling_df [is.na(modeling_df)] <- 0 #have to for standardizing
modeling_df [1:10, 1:10]
Predicted_G1_fulldtm <- as.data.frame (predict(model, newdata = modeling_df))
colnames(Predicted_G1_fulldtm) [1] <- 'RF_fulldtm_pred_G1'
Text_df <- cbind (Text_df, Predicted_G1_fulldtm)
psych::corr.test (Text_df[,c("Ratee_G1_num",  "RF_fulldtm_pred_G1")])

###
### G2 Score ###
###

#Filtering dtm 
dtm_m <- dtm_m_base
cor_df <- dplyr::filter (readRDS("cal_words_fulldtm_CAL.Rda"), Dimension == 'G2')
dim (cor_df)
dtm_m <- dtm_m [,colnames(dtm_m) %in% cor_df[,1]]
ncol(dtm_m)

#creating empty variables for terms that did not appear in your new local file #
add_ons <- list()
for (word in cor_df [,1]) {
  ifelse (word %in% colnames(dtm_m), NA, print(word))
  ifelse (word %in% colnames(dtm_m), NA, add_ons[[word]] <- word)
}
Blank_Columns  <-   data.frame(matrix(0, nrow = nrow(dtm_m), ncol = length(add_ons)))
colnames (Blank_Columns) <- add_ons
dtm_m <- cbind (dtm_m, Blank_Columns)
dtm_m <- dtm_m[,order(colnames(dtm_m))] #ordering so same for algorithms

#creating scores#
model <- readRDS("RF_fit_fulldtm_G2_overallCAL.Rda")
modeling_df <- dtm_m
colnames(modeling_df) <- paste(colnames(modeling_df), "_rf", sep = "")
colnames(modeling_df) <- gsub(" ", "__", colnames(modeling_df))
modeling_df <- as.data.frame (apply (modeling_df, 2, scale)) #normalizing#
modeling_df [is.na(modeling_df)] <- 0 #have to for standardizing
modeling_df [1:10, 1:10]
Predicted_G2_fulldtm <- as.data.frame (predict(model, newdata = modeling_df))
colnames(Predicted_G2_fulldtm) [1] <- 'RF_fulldtm_pred_G2'
Text_df <- cbind (Text_df, Predicted_G2_fulldtm)
psych::corr.test (Text_df[,c("Ratee_G2_num",  "RF_fulldtm_pred_G2")])

###
### G3 Score ###
###

#Filtering dtm 
dtm_m <- dtm_m_base
cor_df <- dplyr::filter (readRDS("cal_words_fulldtm_CAL.Rda"), Dimension == 'G3')
dim (cor_df)
dtm_m <- dtm_m [,colnames(dtm_m) %in% cor_df[,1]]
ncol(dtm_m)

#creating empty variables for terms that did not appear in your new local file #
add_ons <- list()
for (word in cor_df [,1]) {
  ifelse (word %in% colnames(dtm_m), NA, print(word))
  ifelse (word %in% colnames(dtm_m), NA, add_ons[[word]] <- word)
}
Blank_Columns  <-   data.frame(matrix(0, nrow = nrow(dtm_m), ncol = length(add_ons)))
colnames (Blank_Columns) <- add_ons
dtm_m <- cbind (dtm_m, Blank_Columns)
dtm_m <- dtm_m[,order(colnames(dtm_m))] #ordering so same for algorithms

#creating scores#
model <- readRDS("RF_fit_fulldtm_G3_overallCAL.Rda")
modeling_df <- dtm_m
colnames(modeling_df) <- paste(colnames(modeling_df), "_rf", sep = "")
colnames(modeling_df) <- gsub(" ", "__", colnames(modeling_df))
modeling_df <- as.data.frame (apply (modeling_df, 2, scale)) #normalizing#
modeling_df [is.na(modeling_df)] <- 0 #have to for standardizing
modeling_df [1:10, 1:10]
Predicted_G3_fulldtm <- as.data.frame (predict(model, newdata = modeling_df))
colnames(Predicted_G3_fulldtm) [1] <- 'RF_fulldtm_pred_G3'
Text_df <- cbind (Text_df, Predicted_G3_fulldtm)
psych::corr.test (Text_df[,c("Ratee_G3_num",  "RF_fulldtm_pred_G3")])

###
### G4 Score ###
###

#Filtering dtm 
dtm_m <- dtm_m_base
cor_df <- dplyr::filter (readRDS("cal_words_fulldtm_CAL.Rda"), Dimension == 'G4')
dim (cor_df)
dtm_m <- dtm_m [,colnames(dtm_m) %in% cor_df[,1]]
ncol(dtm_m)

#creating empty variables for terms that did not appear in your new local file #
add_ons <- list()
for (word in cor_df [,1]) {
  ifelse (word %in% colnames(dtm_m), NA, print(word))
  ifelse (word %in% colnames(dtm_m), NA, add_ons[[word]] <- word)
}
Blank_Columns  <-   data.frame(matrix(0, nrow = nrow(dtm_m), ncol = length(add_ons)))
colnames (Blank_Columns) <- add_ons
dtm_m <- cbind (dtm_m, Blank_Columns)
dtm_m <- dtm_m[,order(colnames(dtm_m))] #ordering so same for algorithms

#creating scores#
model <- readRDS("RF_fit_fulldtm_G4_overallCAL.Rda")
modeling_df <- dtm_m
colnames(modeling_df) <- paste(colnames(modeling_df), "_rf", sep = "")
colnames(modeling_df) <- gsub(" ", "__", colnames(modeling_df))
modeling_df <- as.data.frame (apply (modeling_df, 2, scale)) #normalizing#
modeling_df [is.na(modeling_df)] <- 0 #have to for standardizing
modeling_df [1:10, 1:10]
Predicted_G4_fulldtm <- as.data.frame (predict(model, newdata = modeling_df))
colnames(Predicted_G4_fulldtm) [1] <- 'RF_fulldtm_pred_G4'
Text_df <- cbind (Text_df, Predicted_G4_fulldtm)
psych::corr.test (Text_df[,c("Ratee_G4_num",  "RF_fulldtm_pred_G4")])

###
### G5 Score ###
###

#Filtering dtm 
dtm_m <- dtm_m_base
cor_df <- dplyr::filter (readRDS("cal_words_fulldtm_CAL.Rda"), Dimension == 'G5')
dim (cor_df)
dtm_m <- dtm_m [,colnames(dtm_m) %in% cor_df[,1]]
ncol(dtm_m)

#creating empty variables for terms that did not appear in your new local file #
add_ons <- list()
for (word in cor_df [,1]) {
  ifelse (word %in% colnames(dtm_m), NA, print(word))
  ifelse (word %in% colnames(dtm_m), NA, add_ons[[word]] <- word)
}
Blank_Columns  <-   data.frame(matrix(0, nrow = nrow(dtm_m), ncol = length(add_ons)))
colnames (Blank_Columns) <- add_ons
dtm_m <- cbind (dtm_m, Blank_Columns)
dtm_m <- dtm_m[,order(colnames(dtm_m))] #ordering so same for algorithms

#creating scores#
model <- readRDS("RF_fit_fulldtm_G5_overallCAL.Rda")
modeling_df <- dtm_m
colnames(modeling_df) <- paste(colnames(modeling_df), "_rf", sep = "")
colnames(modeling_df) <- gsub(" ", "__", colnames(modeling_df))
modeling_df <- as.data.frame (apply (modeling_df, 2, scale)) #normalizing#
modeling_df [is.na(modeling_df)] <- 0 #have to for standardizing
modeling_df [1:10, 1:10]
Predicted_G5_fulldtm <- as.data.frame (predict(model, newdata = modeling_df))
colnames(Predicted_G5_fulldtm) [1] <- 'RF_fulldtm_pred_G5'
Text_df <- cbind (Text_df, Predicted_G5_fulldtm)
psych::corr.test (Text_df[,c("Ratee_G5_num",  "RF_fulldtm_pred_G5")])

###
### G6 Score ###
###

#Filtering dtm 
dtm_m <- dtm_m_base
cor_df <- dplyr::filter (readRDS("cal_words_fulldtm_CAL.Rda"), Dimension == 'G6')
dim (cor_df)
dtm_m <- dtm_m [,colnames(dtm_m) %in% cor_df[,1]]
ncol(dtm_m)

#creating empty variables for terms that did not appear in your new local file #
add_ons <- list()
for (word in cor_df [,1]) {
  ifelse (word %in% colnames(dtm_m), NA, print(word))
  ifelse (word %in% colnames(dtm_m), NA, add_ons[[word]] <- word)
}
Blank_Columns  <-   data.frame(matrix(0, nrow = nrow(dtm_m), ncol = length(add_ons)))
colnames (Blank_Columns) <- add_ons
dtm_m <- cbind (dtm_m, Blank_Columns)
dtm_m <- dtm_m[,order(colnames(dtm_m))] #ordering so same for algorithms

#creating scores#
model <- readRDS("RF_fit_fulldtm_G6_overallCAL.Rda")
modeling_df <- dtm_m
colnames(modeling_df) <- paste(colnames(modeling_df), "_rf", sep = "")
colnames(modeling_df) <- gsub(" ", "__", colnames(modeling_df))
modeling_df <- as.data.frame (apply (modeling_df, 2, scale)) #normalizing#
modeling_df [is.na(modeling_df)] <- 0 #have to for standardizing
modeling_df [1:10, 1:10]
Predicted_G6_fulldtm <- as.data.frame (predict(model, newdata = modeling_df))
colnames(Predicted_G6_fulldtm) [1] <- 'RF_fulldtm_pred_G6'
Text_df <- cbind (Text_df, Predicted_G6_fulldtm)
psych::corr.test (Text_df[,c("Ratee_G6_num",  "RF_fulldtm_pred_G6")])

###
### G7 Score ###
###

#Filtering dtm 
dtm_m <- dtm_m_base
cor_df <- dplyr::filter (readRDS("cal_words_fulldtm_CAL.Rda"), Dimension == 'G7')
dim (cor_df)
dtm_m <- dtm_m [,colnames(dtm_m) %in% cor_df[,1]]
ncol(dtm_m)

#creating empty variables for terms that did not appear in your new local file #
add_ons <- list()
for (word in cor_df [,1]) {
  ifelse (word %in% colnames(dtm_m), NA, print(word))
  ifelse (word %in% colnames(dtm_m), NA, add_ons[[word]] <- word)
}
Blank_Columns  <-   data.frame(matrix(0, nrow = nrow(dtm_m), ncol = length(add_ons)))
colnames (Blank_Columns) <- add_ons
dtm_m <- cbind (dtm_m, Blank_Columns)
dtm_m <- dtm_m[,order(colnames(dtm_m))] #ordering so same for algorithms

#creating scores#
model <- readRDS("RF_fit_fulldtm_G7_overallCAL.Rda")
modeling_df <- dtm_m
colnames(modeling_df) <- paste(colnames(modeling_df), "_rf", sep = "")
colnames(modeling_df) <- gsub(" ", "__", colnames(modeling_df))
modeling_df <- as.data.frame (apply (modeling_df, 2, scale)) #normalizing#
modeling_df [is.na(modeling_df)] <- 0 #have to for standardizing
modeling_df [1:10, 1:10]
Predicted_G7_fulldtm <- as.data.frame (predict(model, newdata = modeling_df))
colnames(Predicted_G7_fulldtm) [1] <- 'RF_fulldtm_pred_G7'
Text_df <- cbind (Text_df, Predicted_G7_fulldtm)
psych::corr.test (Text_df[,c("Ratee_G7_num",  "RF_fulldtm_pred_G7")])

###
### G8 Score ###
###

#Filtering dtm 
dtm_m <- dtm_m_base
cor_df <- dplyr::filter (readRDS("cal_words_fulldtm_CAL.Rda"), Dimension == 'G8')
dim (cor_df)
dtm_m <- dtm_m [,colnames(dtm_m) %in% cor_df[,1]]
ncol(dtm_m)

#creating empty variables for terms that did not appear in your new local file #
add_ons <- list()
for (word in cor_df [,1]) {
  ifelse (word %in% colnames(dtm_m), NA, print(word))
  ifelse (word %in% colnames(dtm_m), NA, add_ons[[word]] <- word)
}
Blank_Columns  <-   data.frame(matrix(0, nrow = nrow(dtm_m), ncol = length(add_ons)))
colnames (Blank_Columns) <- add_ons
dtm_m <- cbind (dtm_m, Blank_Columns)
dtm_m <- dtm_m[,order(colnames(dtm_m))] #ordering so same for algorithms

#creating scores#
model <- readRDS("RF_fit_fulldtm_G8_overallCAL.Rda")
modeling_df <- dtm_m
colnames(modeling_df) <- paste(colnames(modeling_df), "_rf", sep = "")
colnames(modeling_df) <- gsub(" ", "__", colnames(modeling_df))
modeling_df <- as.data.frame (apply (modeling_df, 2, scale)) #normalizing#
modeling_df [is.na(modeling_df)] <- 0 #have to for standardizing
modeling_df [1:10, 1:10]
Predicted_G8_fulldtm <- as.data.frame (predict(model, newdata = modeling_df))
colnames(Predicted_G8_fulldtm) [1] <- 'RF_fulldtm_pred_G8'
Text_df <- cbind (Text_df, Predicted_G8_fulldtm)
psych::corr.test (Text_df[,c("Ratee_G8_num",  "RF_fulldtm_pred_G8")])

###
### overall Score ###
###

#Filtering dtm 
dtm_m <- dtm_m_base
cor_df <- dplyr::filter (readRDS("cal_words_fulldtm_CAL.Rda"), Dimension == 'overall')
dim (cor_df)
dtm_m <- dtm_m [,colnames(dtm_m) %in% cor_df[,1]]
ncol(dtm_m)

#creating empty variables for terms that did not appear in your new local file #
add_ons <- list()
for (word in cor_df [,1]) {
  ifelse (word %in% colnames(dtm_m), NA, print(word))
  ifelse (word %in% colnames(dtm_m), NA, add_ons[[word]] <- word)
}
Blank_Columns  <-   data.frame(matrix(0, nrow = nrow(dtm_m), ncol = length(add_ons)))
colnames (Blank_Columns) <- add_ons
dtm_m <- cbind (dtm_m, Blank_Columns)
dtm_m <- dtm_m[,order(colnames(dtm_m))] #ordering so same for algorithms

#creating scores#
model <- readRDS("RF_fit_fulldtm_overall_overallCA.Rda")
modeling_df <- dtm_m
colnames(modeling_df) <- paste(colnames(modeling_df), "_rf", sep = "")
colnames(modeling_df) <- gsub(" ", "__", colnames(modeling_df))
modeling_df <- as.data.frame (apply (modeling_df, 2, scale)) #normalizing#
modeling_df [is.na(modeling_df)] <- 0 #have to for standardizing
modeling_df [1:10, 1:10]
Predicted_overall_fulldtm <- as.data.frame (predict(model, newdata = modeling_df))
colnames(Predicted_overall_fulldtm) [1] <- 'RF_fulldtm_pred_overall'
Text_df <- cbind (Text_df, Predicted_overall_fulldtm)
psych::corr.test (Text_df[,c("Performance_Composite",  "RF_fulldtm_pred_overall")])

###
### Combining Scores: fulldtm ###
###
fulldtm_Scores <- cbind (Predicted_G1_fulldtm, Predicted_G2_fulldtm, Predicted_G3_fulldtm, Predicted_G4_fulldtm, Predicted_G5_fulldtm, Predicted_G6_fulldtm, 
                         Predicted_G7_fulldtm, Predicted_G8_fulldtm, Predicted_overall_fulldtm)
colnames(fulldtm_Scores)
psych::corr.test(fulldtm_Scores)
dim(fulldtm_Scores)
saveRDS(fulldtm_Scores, "fulldtm_Scores.Rda")

############################################################################################################################################
############################################################################################################################################
#################################################  Creating Scores: Global Word Embeddings  ###################################################
############################################################################################################################################
############################################################################################################################################

Text_df_Embed$Word_count <- str_count(Text_df_Embed$Narrative, "\\w+")
Text_df_Embed$Characters <- nchar (Text_df_Embed$Narrative)

count <- Text_df_Embed$Word_count
file <- dplyr::select(Text_df_Embed, Unique_ID_Ratee, Narrative)
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
dtm_ordered <- dtm  [order(rownames(dtm)),]    #ordering variables
ID_var <- as.data.frame (rownames(dtm)) #creating ID variable for matching later
names(ID_var) [1] <- 'Unique_ID_Ratee' #creating ID variable for matching later
dtm <- as.matrix (apply (dtm, 2, scale)) #normalizing
dtm [1:10, 1:10]
dim (dtm)

### storing possible words so that I can filter down the GloVe embedding file ###
names <- as.data.frame (colnames(dtm))
colnames(names)[1] <- 'names'

### Pulling in Glove Data ###
#To begin, download one of the GloVe files with 300 dimensions from https://nlp.stanford.edu/projects/glove/, run by Pennington, Socher, & Manning. 
#I'm using the "glove.42B.300d" file. It should be saved as "glove.42B.300d.txt within your local folder"
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
g6b_300 <- scan(file = "glove.42B.300d.txt", what="", sep="\n")
# call the function to convert the raw GloVe vector to data.frame
glove.300 <- proc_pretrained_vec(g6b_300)  # this is the actual function call
dim (glove.300)

#### Filtering that down and scoring the vectors on local data ###
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

#### Scoring and Finishing Setup ###
embeddings <- as.data.frame (as.matrix (dtm %*% glove.300)) #uses product of dtm and embedding matrix, average of vectors of words#
dim(embeddings) #N by k (300)
embeddings <- embeddings / count #controlling for number of words#
embeddings <- cbind (ID_var, embeddings)
embeddings <- merge (dplyr::select(Text_df_Embed, Unique_ID_Ratee, Word_count), embeddings, by = "Unique_ID_Ratee", all=TRUE)

vars <- embeddings [,2:ncol(embeddings)]
vars <- as.data.frame (apply (vars, 2, scale)) #normalizing
embeddings <- cbind (dplyr::select(embeddings, Unique_ID_Ratee), vars)
embeddings [1:10, 1:10]
glove.300 <- NULL

#####
##### Scoring G1 ###
#####
Glove_Model_G1 <- readRDS ("RF_fit_glove_G1_CAL.Rda")
Predicted_Glove_G1 <- as.data.frame (predict(Glove_Model_G1, newdata = embeddings))
colnames(Predicted_Glove_G1) [1] <- 'RF_glove_pred_G1'
file <- dplyr::select(embeddings, Unique_ID_Ratee)
Predicted_Glove_G1 <- cbind (Predicted_Glove_G1, file)
Text_df_Embed <- merge (Text_df_Embed, Predicted_Glove_G1, by = "Unique_ID_Ratee", all.x=TRUE)
corr.test(dplyr::select(Text_df_Embed, RF_glove_pred_G1, Ratee_G1_num))

#####
##### Scoring G2 ###
#####
Glove_Model_G2 <- readRDS ("RF_fit_glove_G2_CAL.Rda")
Predicted_Glove_G2 <- as.data.frame (predict(Glove_Model_G2, newdata = embeddings))
colnames(Predicted_Glove_G2) [1] <- 'RF_glove_pred_G2'
file <- dplyr::select(embeddings, Unique_ID_Ratee)
Predicted_Glove_G2 <- cbind (Predicted_Glove_G2, file)
Text_df_Embed <- merge (Text_df_Embed, Predicted_Glove_G2, by = "Unique_ID_Ratee", all.x=TRUE)
corr.test(dplyr::select(Text_df_Embed, RF_glove_pred_G2, Ratee_G2_num))

#####
##### Scoring G3 ###
#####
Glove_Model_G3 <- readRDS ("RF_fit_glove_G3_CAL.Rda")
Predicted_Glove_G3 <- as.data.frame (predict(Glove_Model_G3, newdata = embeddings))
colnames(Predicted_Glove_G3) [1] <- 'RF_glove_pred_G3'
file <- dplyr::select(embeddings, Unique_ID_Ratee)
Predicted_Glove_G3 <- cbind (Predicted_Glove_G3, file)
Text_df_Embed <- merge (Text_df_Embed, Predicted_Glove_G3, by = "Unique_ID_Ratee", all.x=TRUE)
corr.test(dplyr::select(Text_df_Embed, RF_glove_pred_G3, Ratee_G3_num))

#####
##### Scoring G4 ###
#####
Glove_Model_G4 <- readRDS ("RF_fit_glove_G4_CAL.Rda")
Predicted_Glove_G4 <- as.data.frame (predict(Glove_Model_G4, newdata = embeddings))
colnames(Predicted_Glove_G4) [1] <- 'RF_glove_pred_G4'
file <- dplyr::select(embeddings, Unique_ID_Ratee)
Predicted_Glove_G4 <- cbind (Predicted_Glove_G4, file)
Text_df_Embed <- merge (Text_df_Embed, Predicted_Glove_G4, by = "Unique_ID_Ratee", all.x=TRUE)
corr.test(dplyr::select(Text_df_Embed, RF_glove_pred_G4, Ratee_G4_num))

#####
##### Scoring G5 ###
#####
Glove_Model_G5 <- readRDS ("RF_fit_glove_G5_CAL.Rda")
Predicted_Glove_G5 <- as.data.frame (predict(Glove_Model_G5, newdata = embeddings))
colnames(Predicted_Glove_G5) [1] <- 'RF_glove_pred_G5'
file <- dplyr::select(embeddings, Unique_ID_Ratee)
Predicted_Glove_G5 <- cbind (Predicted_Glove_G5, file)
Text_df_Embed <- merge (Text_df_Embed, Predicted_Glove_G5, by = "Unique_ID_Ratee", all.x=TRUE)
corr.test(dplyr::select(Text_df_Embed, RF_glove_pred_G5, Ratee_G5_num))

#####
##### Scoring G6 ###
#####
Glove_Model_G6 <- readRDS ("RF_fit_glove_G6_CAL.Rda")
Predicted_Glove_G6 <- as.data.frame (predict(Glove_Model_G6, newdata = embeddings))
colnames(Predicted_Glove_G6) [1] <- 'RF_glove_pred_G6'
file <- dplyr::select(embeddings, Unique_ID_Ratee)
Predicted_Glove_G6 <- cbind (Predicted_Glove_G6, file)
Text_df_Embed <- merge (Text_df_Embed, Predicted_Glove_G6, by = "Unique_ID_Ratee", all.x=TRUE)
corr.test(dplyr::select(Text_df_Embed, RF_glove_pred_G6, Ratee_G6_num))

#####
##### Scoring G7 ###
#####
Glove_Model_G7 <- readRDS ("RF_fit_glove_G7_CAL.Rda")
Predicted_Glove_G7 <- as.data.frame (predict(Glove_Model_G7, newdata = embeddings))
colnames(Predicted_Glove_G7) [1] <- 'RF_glove_pred_G7'
file <- dplyr::select(embeddings, Unique_ID_Ratee)
Predicted_Glove_G7 <- cbind (Predicted_Glove_G7, file)
Text_df_Embed <- merge (Text_df_Embed, Predicted_Glove_G7, by = "Unique_ID_Ratee", all.x=TRUE)
corr.test(dplyr::select(Text_df_Embed, RF_glove_pred_G7, Ratee_G7_num))

#####
##### Scoring G8 ###
#####
Glove_Model_G8 <- readRDS ("RF_fit_glove_G8_CAL.Rda")
Predicted_Glove_G8 <- as.data.frame (predict(Glove_Model_G8, newdata = embeddings))
colnames(Predicted_Glove_G8) [1] <- 'RF_glove_pred_G8'
file <- dplyr::select(embeddings, Unique_ID_Ratee)
Predicted_Glove_G8 <- cbind (Predicted_Glove_G8, file)
Text_df_Embed <- merge (Text_df_Embed, Predicted_Glove_G8, by = "Unique_ID_Ratee", all.x=TRUE)
corr.test(dplyr::select(Text_df_Embed, RF_glove_pred_G8, Ratee_G8_num))

#####
##### Scoring overall ###
#####
Glove_Model_overall <- readRDS ("RF_fit_glove_overall_CAL.Rda")
Predicted_Glove_overall <- as.data.frame (predict(Glove_Model_overall, newdata = embeddings))
colnames(Predicted_Glove_overall) [1] <- 'RF_glove_pred_overall'
file <- dplyr::select(embeddings, Unique_ID_Ratee)
Predicted_Glove_overall <- cbind (Predicted_Glove_overall, file)
Text_df_Embed <- merge (Text_df_Embed, Predicted_Glove_overall, by = "Unique_ID_Ratee", all.x=TRUE)
corr.test(dplyr::select(Text_df_Embed, RF_glove_pred_overall, Performance_Composite))

###
### Combining Scores: Glove ###
###
Glove_Scores <- merge (Predicted_Glove_G1, Predicted_Glove_G2, by = "Unique_ID_Ratee", all.x=TRUE)
Glove_Scores <- merge (Glove_Scores, Predicted_Glove_G3, by = "Unique_ID_Ratee", all.x=TRUE)
Glove_Scores <- merge (Glove_Scores, Predicted_Glove_G4, by = "Unique_ID_Ratee", all.x=TRUE)
Glove_Scores <- merge (Glove_Scores, Predicted_Glove_G5, by = "Unique_ID_Ratee", all.x=TRUE)
Glove_Scores <- merge (Glove_Scores, Predicted_Glove_G6, by = "Unique_ID_Ratee", all.x=TRUE)
Glove_Scores <- merge (Glove_Scores, Predicted_Glove_G7, by = "Unique_ID_Ratee", all.x=TRUE)
Glove_Scores <- merge (Glove_Scores, Predicted_Glove_G8, by = "Unique_ID_Ratee", all.x=TRUE)
Glove_Scores <- merge (Glove_Scores, Predicted_Glove_overall, by = "Unique_ID_Ratee", all.x=TRUE)
colnames(Glove_Scores)
psych::corr.test(Glove_Scores)
dim(Glove_Scores)
saveRDS(Glove_Scores, "Glove_Scores.Rda")

############################################################################################################################################
############################################################################################################################################
#################################################  Combining Scores and Norming  ###################################################
############################################################################################################################################
############################################################################################################################################

### Loading original file and newly created scores ###
Text_df <- readRDS ("Text_df.Rda") #if you didn't name the file "Text_df", insert the name of original file
Theme_Scores <- readRDS ("Theme_Scores.Rda")
Ngram_Scores <- readRDS ("Ngram_Scores.Rda")
fulldtm_Scores <- readRDS("fulldtm_Scores.Rda")
Glove_Scores <- readRDS ("Glove_Scores.Rda")

Text_df <- cbind (Text_df, Theme_Scores, Ngram_Scores, fulldtm_Scores)
Text_df <- merge (Text_df, Glove_Scores, by = "Unique_ID_Ratee")
colnames(Text_df)
dim (Text_df)

### Loading in norms; these are average raw values across the Great 8 dimensions, taken from Study 1
Norms_Theme <- readRDS("Norms_Theme.Rda")
Norms_Valence <- readRDS("Norms_Valence.Rda")

# Norming and transforming theme scores #
Text_df$G1_Theme_Score_AtallYesNo <- ifelse (Text_df$G1_Theme_Score > 0, 1, 0)
Text_df$G2_Theme_Score_AtallYesNo <- ifelse (Text_df$G2_Theme_Score > 0, 1, 0)
Text_df$G3_Theme_Score_AtallYesNo <- ifelse (Text_df$G3_Theme_Score > 0, 1, 0)
Text_df$G4_Theme_Score_AtallYesNo <- ifelse (Text_df$G4_Theme_Score > 0, 1, 0)
Text_df$G5_Theme_Score_AtallYesNo <- ifelse (Text_df$G5_Theme_Score > 0, 1, 0)
Text_df$G6_Theme_Score_AtallYesNo <- ifelse (Text_df$G6_Theme_Score > 0, 1, 0)
Text_df$G7_Theme_Score_AtallYesNo <- ifelse (Text_df$G7_Theme_Score > 0, 1, 0)
Text_df$G8_Theme_Score_AtallYesNo <- ifelse (Text_df$G8_Theme_Score > 0, 1, 0)
table (Text_df$G1_Theme_Score_AtallYesNo)

Text_df$G1_Theme_Score <- (((Text_df$G1_Theme_Score - Norms_Theme[1,1])/ Norms_Theme[2,1]) * 10) + 100
Text_df$G2_Theme_Score <- (((Text_df$G2_Theme_Score - Norms_Theme[1,1])/ Norms_Theme[2,1]) * 10) + 100
Text_df$G3_Theme_Score <- (((Text_df$G3_Theme_Score - Norms_Theme[1,1])/ Norms_Theme[2,1]) * 10) + 100
Text_df$G4_Theme_Score <- (((Text_df$G4_Theme_Score - Norms_Theme[1,1])/ Norms_Theme[2,1]) * 10) + 100
Text_df$G5_Theme_Score <- (((Text_df$G5_Theme_Score - Norms_Theme[1,1])/ Norms_Theme[2,1]) * 10) + 100
Text_df$G6_Theme_Score <- (((Text_df$G6_Theme_Score - Norms_Theme[1,1])/ Norms_Theme[2,1]) * 10) + 100
Text_df$G7_Theme_Score <- (((Text_df$G7_Theme_Score - Norms_Theme[1,1])/ Norms_Theme[2,1]) * 10) + 100
Text_df$G8_Theme_Score <- (((Text_df$G8_Theme_Score - Norms_Theme[1,1])/ Norms_Theme[2,1]) * 10) + 100
psych::describe (dplyr::select(Text_df, G1_Theme_Score:G8_Theme_Score))

# Creating Composite Valence Scores based on norms #
#first, Z-scoring based on study norms and combining scores into composite
Text_df$Valence_G1 <- ((Text_df$RF_ngram_pred_G1-Norms_Valence[1,1])/ Norms_Valence[2,1]) + ((Text_df$RF_glove_pred_G1-Norms_Valence[3,1])/Norms_Valence[4,1]) + 
  ((Text_df$RF_fulldtm_pred_G1 -Norms_Valence[5,1])/Norms_Valence[6,1])
Text_df$Valence_G2 <- ((Text_df$RF_ngram_pred_G2-Norms_Valence[1,1])/ Norms_Valence[2,1]) + ((Text_df$RF_glove_pred_G2-Norms_Valence[3,1])/Norms_Valence[4,1]) + 
  ((Text_df$RF_fulldtm_pred_G2 -Norms_Valence[5,1])/Norms_Valence[6,1])
Text_df$Valence_G3 <- ((Text_df$RF_ngram_pred_G3-Norms_Valence[1,1])/ Norms_Valence[2,1]) + ((Text_df$RF_glove_pred_G3-Norms_Valence[3,1])/Norms_Valence[4,1]) + 
  ((Text_df$RF_fulldtm_pred_G3 -Norms_Valence[5,1])/Norms_Valence[6,1])
Text_df$Valence_G4 <- ((Text_df$RF_ngram_pred_G4-Norms_Valence[1,1])/ Norms_Valence[2,1]) + ((Text_df$RF_glove_pred_G4-Norms_Valence[3,1])/Norms_Valence[4,1]) + 
  ((Text_df$RF_fulldtm_pred_G4 -Norms_Valence[5,1])/Norms_Valence[6,1])
Text_df$Valence_G5 <- ((Text_df$RF_ngram_pred_G5-Norms_Valence[1,1])/ Norms_Valence[2,1]) + ((Text_df$RF_glove_pred_G5-Norms_Valence[3,1])/Norms_Valence[4,1]) + 
  ((Text_df$RF_fulldtm_pred_G5 -Norms_Valence[5,1])/Norms_Valence[6,1])
Text_df$Valence_G6 <- ((Text_df$RF_ngram_pred_G6-Norms_Valence[1,1])/ Norms_Valence[2,1]) + ((Text_df$RF_glove_pred_G6-Norms_Valence[3,1])/Norms_Valence[4,1]) + 
  ((Text_df$RF_fulldtm_pred_G6 -Norms_Valence[5,1])/Norms_Valence[6,1])
Text_df$Valence_G7 <- ((Text_df$RF_ngram_pred_G7-Norms_Valence[1,1])/ Norms_Valence[2,1]) + ((Text_df$RF_glove_pred_G7-Norms_Valence[3,1])/Norms_Valence[4,1]) + 
  ((Text_df$RF_fulldtm_pred_G7 -Norms_Valence[5,1])/Norms_Valence[6,1])
Text_df$Valence_G8 <- ((Text_df$RF_ngram_pred_G8-Norms_Valence[1,1])/ Norms_Valence[2,1]) + ((Text_df$RF_glove_pred_G8-Norms_Valence[3,1])/Norms_Valence[4,1]) + 
  ((Text_df$RF_fulldtm_pred_G8 -Norms_Valence[5,1])/Norms_Valence[6,1])
Text_df$Valence_Overall <- ((Text_df$RF_ngram_pred_overall-Norms_Valence[1,1])/ Norms_Valence[2,1]) + ((Text_df$RF_glove_pred_overall-Norms_Valence[3,1])/Norms_Valence[4,1]) + 
  ((Text_df$RF_fulldtm_pred_overall -Norms_Valence[5,1])/Norms_Valence[6,1])

Text_df$Valence_G1 <- (Text_df$Valence_G1 - Norms_Valence[7,1])/ Norms_Valence[8,1]
Text_df$Valence_G2 <- (Text_df$Valence_G2 - Norms_Valence[7,1])/ Norms_Valence[8,1]
Text_df$Valence_G3 <- (Text_df$Valence_G3 - Norms_Valence[7,1])/ Norms_Valence[8,1]
Text_df$Valence_G4 <- (Text_df$Valence_G4 - Norms_Valence[7,1])/ Norms_Valence[8,1]
Text_df$Valence_G5 <- (Text_df$Valence_G5 - Norms_Valence[7,1])/ Norms_Valence[8,1]
Text_df$Valence_G6 <- (Text_df$Valence_G6 - Norms_Valence[7,1])/ Norms_Valence[8,1]
Text_df$Valence_G7 <- (Text_df$Valence_G7 - Norms_Valence[7,1])/ Norms_Valence[8,1]
Text_df$Valence_G8 <- (Text_df$Valence_G8 - Norms_Valence[7,1])/ Norms_Valence[8,1]
Text_df$Valence_Overall <- (Text_df$Valence_Overall - Norms_Valence[7,1])/ Norms_Valence[8,1]

# adjusting scale #
Text_df$Valence_G1 <- (Text_df$Valence_G1 * 10) + 100
Text_df$Valence_G2 <- (Text_df$Valence_G2 * 10) + 100
Text_df$Valence_G3 <- (Text_df$Valence_G3 * 10) + 100
Text_df$Valence_G4 <- (Text_df$Valence_G4 * 10) + 100
Text_df$Valence_G5 <- (Text_df$Valence_G5 * 10) + 100
Text_df$Valence_G6 <- (Text_df$Valence_G6 * 10) + 100
Text_df$Valence_G7 <- (Text_df$Valence_G7 * 10) + 100
Text_df$Valence_G8 <- (Text_df$Valence_G8 * 10) + 100
Text_df$Valence_Overall <- (Text_df$Valence_Overall * 10) + 100
psych::describe(dplyr::select(Text_df, Valence_G1:Valence_Overall))

### Saving ###
colnames(Text_df)
saveRDS (Text_df, "Text_df_final.Rda") # That's it! Your file should now have all the newly created scores!


