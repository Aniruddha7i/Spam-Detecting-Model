# -*- coding: utf-8 -*-
"""Spam_sms.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1_5S-7Mmu1DxKNHCmS_W3P96BRgDDmT0V

# Import and loding data
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load data
df = pd.read_csv('spam.csv', encoding='latin-1')
df.head()

# 1. Data cleaning
# 2. EDA
# 3. Text preprocessing
# 4. Model building
# 5. Evaluation
# 6. Improvement

"""# Data Cleaning"""

df.info()

## we need to drop 3, 4, 5 columns because maximun are null values
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

df.sample(5)

# change column name
df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)
df.head()

import sklearn
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

df['target'] = encoder.fit_transform(df['target'])

df.head()
## spam 1 and ham 0

## cheak for duplicate values
df.duplicated().sum()

# there are 403 duplicate values in our dataset
# hence remove it
df = df.drop_duplicates(keep='first')

df.duplicated().sum()
## now there are no duplicate values

# df.shape()

"""# EDA"""

# cheak data balance or not
df['target'].value_counts()

# unequal distribution of data
# 0 4516
# 1 653
# vesulise data distribution
import matplotlib.pyplot as plt

plt.pie(df['target'].value_counts(), labels=['ham', 'spam'], autopct="%0.2f")
plt.show()

# data is imbalance

# creat some new fretures

import nltk

nltk.download('punkt')

# count number of char in text
df['num_char'] = df['text'].apply(len)
print(df['num_char'])

# count number of words in text
df['num_words'] = df['text'].apply(lambda x: len(nltk.word_tokenize(x)))
print(df['num_words'])

df.head()

# count number of sentance in text
df['num_sentance'] = df['text'].apply(lambda x: len(nltk.sent_tokenize(x)))
print(df['num_sentance'])

df.sample(5)

# describe our new fretures
df[['num_char', 'num_words', 'num_sentance']].describe()

df[df['target'] == 0][['num_char', 'num_words', 'num_sentance']].describe()

df[df['target'] == 1][['num_char', 'num_words', 'num_sentance']].describe()

# histogram plot of char
import seaborn as sns

sns.histplot(df[df['target'] == 0]['num_char'])

sns.histplot(df[df['target'] == 1]['num_char'])

# for ham max char are 0 to 100 char per sms
# for spam max char are in between 100 to 150
sns.histplot(df[df['target'] == 0]['num_char'], color='green')
sns.histplot(df[df['target'] == 1]['num_char'], color='red')

sns.histplot(df[df['target'] == 0]['num_words'], color='green')
sns.histplot(df[df['target'] == 1]['num_words'], color='red')

# sns.histplot(df[df['target']==0]['num_sentance'],color='green')
# sns.histplot(df[df['target']==1]['num_sentance'],color='red')

sns.pairplot(df, hue='target')

# there are so many outliears

# find coleration cofficents
df['num_words'] = df['num_words'].astype(int)
df['num_sentance'] = df['num_sentance'].astype(int)
df['num_char'] = df['num_char'].astype(int)

# Select only numerical columns before calculating correlations
numerical_df = df.select_dtypes(include=['number'])
correlation_matrix = numerical_df.corr()
print(correlation_matrix)

sns.heatmap(correlation_matrix, annot=True)

# Multi colinearity is present
# hence we can take only one fretures
# we take num_char column

"""# **Data Preprocessing**"""

#### Text preprocessing
# 1. Lower case
# 2. Tokenization
# 3. Removing special characters
# 4. Removing stop words and punctuation
# 5. Stemming

import nltk

nltk.download('stopwords')

stopWords = nltk.corpus.stopwords.words('english')
# those words which are not need for sentance formation

import string


# string.punctuation

def transform_text(text):
    #convart lower case
    text = text.lower()

    # tokenization :: convart string in to a list of words
    text = nltk.word_tokenize(text)

    # remove spatial char
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]  # this is cloning
    y.clear()

    # remove stop words and punctuation
    for i in text:
        if i not in stopWords and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    # stemming
    # stemming is process of reducing a word to its root word
    # for example danceing ---> dance
    ps = nltk.porter.PorterStemmer()
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


transform_text("Hello, &bn20 how are you?")

df['transformed_text'] = df['text'].apply(transform_text)

df.sample(5)

## word cloud
# from wordcloud import WordCloud
# wc = WordCloud(width=500,height=500,min_font_size=10,background_color='black')
#
# spam_wc = wc.generate(df[df['target']==1]['transformed_text'].str.cat(sep=" "))
#
# plt.figure(figsize=(14,5))
# plt.imshow(spam_wc)
#
# ham_wc = wc.generate(df[df['target']==0]['transformed_text'].str.cat(sep=" "))
#
# plt.figure(figsize=(14,5))
# plt.imshow(ham_wc)

# spam_corpus=[]
# for msg in df[df['target']==1]['transformed_text'].tolist():
#   for word in msg.split():
#     spam_corpus.append(word)
#
# len(spam_corpus)
#
# from collections import Counter
# sns.barplot(x=pd.DataFrame(Counter(spam_corpus).most_common(30))[0], y=pd.DataFrame(Counter(spam_corpus).most_common(30))[1])
# plt.xticks(rotation='vertical')
# plt.show()
#
# ham_corpus=[]
# for msg in df[df['target']==0]['transformed_text'].tolist():
#   for word in msg.split():
#     ham_corpus.append(word)
#
# len(ham_corpus)
#
# from collections import Counter
# sns.barplot(x=pd.DataFrame(Counter(ham_corpus).most_common(30))[0], y=pd.DataFrame(Counter(ham_corpus).most_common(30))[1])
# plt.xticks(rotation='vertical')
# plt.show()
#


"""# Modeling"""

df.sample(5)

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

cv = CountVectorizer()
x = cv.fit_transform(df['transformed_text']).toarray()

"""Only transformed_text as a fretures and NB **molel**"""

# x.shape

print(x)

y = df['target'].values

## split data
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

## here we use all possible NB classes
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import CategoricalNB

# NB_Gaussian = GaussianNB()
# NB_Multinomial = MultinomialNB()
# NB_Bernoulli = BernoulliNB()
# NB_Complement = ComplementNB()
# NB_Categorical = CategoricalNB()
#
# ## fit the modale
# NB_Gaussian.fit(x_train, y_train)
# NB_Multinomial.fit(x_train, y_train)
# NB_Bernoulli.fit(x_train, y_train)
# NB_Complement.fit(x_train, y_train)
# NB_Categorical.fit(x_train, y_train)
#
# ## predect data
# y_Gaussian = NB_Gaussian.predict(x_test);
# y_Multinomial = NB_Multinomial.predict(x_test);
# y_Bernoulli = NB_Bernoulli.predict(x_test);
# y_Complement = NB_Complement.predict(x_test);
# # y_Categorical = NB_Categorical.predict(x_test);

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score

# gauss_confusion = confusion_matrix(y_test, y_Gaussian)
# print(gauss_confusion)
#
# gauss_acc = accuracy_score(y_test, y_Gaussian)
# print(gauss_acc)
#
# gauss_pre = precision_score(y_test, y_Gaussian)
# print(gauss_pre)
#
# Multi_confusion = confusion_matrix(y_test, y_Multinomial)
# print(Multi_confusion)
#
# Multi_acc = accuracy_score(y_test, y_Multinomial)
# print(Multi_acc)
#
# Multi_pre = precision_score(y_test, y_Multinomial)
# print(Multi_pre)
#
# Bern_confusion = confusion_matrix(y_test, y_Bernoulli)
# print(Bern_confusion)
#
# Bern_acc = accuracy_score(y_test, y_Bernoulli)
# print(Bern_acc)
#
# Bern_pre = precision_score(y_test, y_Bernoulli)
# print(Bern_pre)
#
# comp_confusion = confusion_matrix(y_test, y_Complement)
# print(comp_confusion)
#
# comp_acc = accuracy_score(y_test, y_Complement)
# print(comp_acc)
#
# comp_pre = precision_score(y_test, y_Complement)
# print(comp_pre)

# still now Bernoulli gives us best result

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier

# from xgboost import XGBClassifier

# with out hyperthermia tuning
svc = SVC(kernel='sigmoid', gamma=1.0)
knc = KNeighborsClassifier()
mnb = MultinomialNB()
bnb = BernoulliNB()
dtc = DecisionTreeClassifier(max_depth=5)
lrc = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=50, random_state=2)
abc = AdaBoostClassifier(n_estimators=50, random_state=2)
bc = BaggingClassifier(n_estimators=50, random_state=2)
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
gbdt = GradientBoostingClassifier(n_estimators=50, random_state=2)
# xgb = XGBClassifier(n_estimators=50, random_state=2)

clfs = {
    'SVC': svc,
    'KN': knc,
    'NBM': mnb,
    'NBG': bnb,
    'DT': dtc,
    'LR': lrc,
    'RF': rfc,
    'AdaBoost': abc,
    'BgC': bc,
    'ETC': etc,
    'GBDT': gbdt
    # 'xgb': xgb
}

# def train_classifier(clf, X_train, y_train, X_test, y_test):
#     clf.fit(X_train, y_train)
#     y_pred = clf.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred)
#
#     return accuracy, precision
#
#
# accuracy_scores = []
# precision_scores = []
#
# for name, clf in clfs.items():
#     current_accuracy, current_precision = train_classifier(clf, x_train, y_train, x_test, y_test)
#
#     print("For ", name)
#     print("Accuracy - ", current_accuracy)
#     print("Precision - ", current_precision)
#
#     accuracy_scores.append(current_accuracy)
#     precision_scores.append(current_precision)
#
# # because of unbalance data distribution of ham and spam :: we focuse in precision
# performance_df = pd.DataFrame(
#     {'Algorithm': clfs.keys(), 'Accuracy': accuracy_scores, 'Precision': precision_scores}).sort_values('Precision',
#                                                                                                         ascending=False)
#
# performance_df

# diffrent vectorizer :: TfidfVectorizer
ifid = TfidfVectorizer(max_features=3000);
x2 = ifid.fit_transform(df['transformed_text']).toarray()

x_train2, x_test2, y_train2, y_test2 = train_test_split(x2, y, test_size=0.2, random_state=2)

# accuracy_scores2 = []
# precision_scores2 = []
#
# for name, clf in clfs.items():
#     current_accuracy, current_precision = train_classifier(clf, x_train2, y_train2, x_test2, y_test2)
#
#     # print("For ",name)
#     # print("Accuracy - ",current_accuracy)
#     # print("Precision - ",current_precision)
#
#     accuracy_scores2.append(current_accuracy)
#     precision_scores2.append(current_precision)
#
# performance2_df = pd.DataFrame(
#     {'Algorithm': clfs.keys(), 'Accuracy ifid': accuracy_scores2, 'Precision ifid': precision_scores2}).sort_values(
#     'Precision ifid', ascending=False)
#
# performance_df.merge(performance2_df, on='Algorithm')
#
# performance2_df

# NBM RF NBG NBM are best model

# Esample model
# use x2 and y
rfc = RandomForestClassifier(n_estimators=50, random_state=2)
mnb = MultinomialNB()
bnb = BernoulliNB()
# Voting Classifier
from sklearn.ensemble import VotingClassifier

voting = VotingClassifier(estimators=[('rf', rfc), ('mnb', mnb), ('bnb', bnb)], voting='soft')

voting.fit(x_train2, y_train2)

voting_y = voting.predict(x_test2)

voting_confusion = confusion_matrix(y_test2, voting_y)
print(voting_confusion)

voting_acc = accuracy_score(y_test2, voting_y)
print(voting_acc)

voting_pre = precision_score(y_test2, voting_y)
print(voting_pre)

# voting is last model we use it
mess_pred = "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's,,,"
rr = [mess_pred]
x22 = ifid.transform(rr).toarray()
pp = voting.predict(x22)

print(pp)

# import pickle
#
# with open('model.pkl', 'wb') as file:
#     pickle.dump(voting, file)
#
# with open('ifid.pkl', 'wb') as file2:
#     pickle.dump(ifid, file2)
