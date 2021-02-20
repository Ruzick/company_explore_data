import glob
import pandas as pd
import nltk
from nltk import word_tokenize, sent_tokenize
# from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer, PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
import dill
#used  python -m nltk.downloader stopwords, then saved file to be imported

#################################################################
#
#                                        Make a program that conditions the dataframe
#            -Removes punctuation -Makes everything lower case, Removes Stop Words
#            -Lematizes
#
#################################################################

def conditioner(text_column):
    text_column = text_column.astype(str) #make sure type string
    text_column=text_column.apply(lambda x: " ".join(x.lower() for x in x.split() ))#make lowercase
    text_column=text_column.str.replace('[^\w\s]','') #remove punctuation
    #remove stop words borred from library
    stop= pd.read_csv("english_stop.txt")
    text_column=text_column.apply(lambda x: " ".join(x for x in x.split() if x not in stop ))
    #stemming
    st = PorterStemmer()
    text_column=text_column.apply(lambda x: " ".join([st.stem(word) for word in x.split()] ))
    return  text_column

###################################################################
#                                               
#                                             start importing the company files
#                                                                                                                                           
###################################################################
path2 = r'data' # use your path
all_files2 = glob.glob(path2 + "/*.csv")

li2 = []
li_samplesize=[]
for filename in all_files2:
    
    df = pd.read_csv(filename, index_col=0, header=0)
    df.insert(0, 'company', filename[:-4].replace("data", "").strip('/').replace("-"," ").strip("\\"))
    #df['company'] = str(filename).strip('.csv').strip('data/')
    li2.append(df)
    li_samplesize.append([filename[:-4].replace("data", "").strip('/').replace("-"," ").strip("\\"), df.size ])
frame2 = pd.concat(li2, axis=0, ignore_index=True)
import numpy as np
np.savetxt("li_samplesize.csv",  
           li_samplesize, 
           delimiter =", ",  
           fmt ='% s')
################################################################
# 
#                                      Only keep company and ratings
#
###############################################################
df1=frame2['company']
df2=frame2['pros']
pos=pd.concat([df1, df2], axis=1, join="inner")
pos.columns=['company', 'text']
neg=pd.concat([df1, frame2['cons']], axis=1, join="inner")
neg.columns=['company', 'text']
frames=(pos, neg)
db = pd.concat(frames, axis=0, ignore_index=False)
condi = conditioner(db['text'])
##################################################################
#
#                                               Labeling
#                       
#################################################################



#########################################################
#
#                  Let's train our model using Amazon's reviews on glassdoor
#
###############################################################

#Lets build our database
#import files to be trained, we will 6,000 reviews total: 3,000 + and 3,000 -

path2 = r'data' # use your path
all_files2 = glob.glob(path2 + "/*.csv")

sentiment_dat = []

for i in range(3):
    
    df = pd.read_csv(all_files2[i], index_col=0, header=0)
    df.insert(0, 'company', all_files2[i][:-4].replace("data", "").strip('/').replace("-"," ").strip("\\"))
    #df['company'] = str(filename).strip('.csv').strip('data/')
    sentiment_dat.append(df)

frame2 = pd.concat(sentiment_dat, axis=0, ignore_index=True)

#conditioning the data
#All the Pros get assigned to positive review of label +1, all the cons to negative review of label -1

#Condition the data 
positive = pd.DataFrame(frame2["pros"].values)
negative = pd.DataFrame(frame2["cons"].values)
positive = positive.rename(columns = {"pros": "text"})
negative = negative.rename(columns = {"cons": "text"})
positive['sentiment'] = 1
negative['sentiment'] = -1 #changed to -1 to be used later in clustering
positive = positive.replace(r'\n','', regex=True) 
negative = negative.replace(r'\n','', regex=True) 
# labeling has been decided  depending of cons or pros
frames=[positive, negative]
reviews=pd.concat(frames, axis=0, ignore_index=False)
all_reviews = pd.concat(frames, axis=0, ignore_index=False)
all_reviews=all_reviews.rename(columns = {0: "text"})
##
#Apply conditioner function
##
all_reviews['text']=conditioner(all_reviews['text'])
reviews = (all_reviews['text'].values)
labels = (all_reviews['sentiment'].values)


from sklearn.model_selection import train_test_split

text_train, text_test, sentiment_train, sentiment_test = train_test_split(reviews, labels, test_size=0.2, random_state=45)
text_train,text_val, sentiment_train, sentiment_val = train_test_split(text_train, sentiment_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2

from sklearn.pipeline import Pipeline
vectorizer=CountVectorizer()
clf= LogisticRegression()
vec_clf = Pipeline([('countV', vectorizer), ('svm', clf)])
vec_clf.fit(text_train,sentiment_train)


# new_review=["this company needs to make a change asap, their cafeteria food is always old", "yeah its fabulous, they sure know how to make you feel at home, great view of the city", "inclusion and diversity are strong "]
# # X_new=vectorizer.transform(new_review) no need to vectorize it anymore, is in pipeline
# lab=vec_clf.predict(new_review)
# print(new_review,lab)





labels=[]
for review in condi:
    labels.append(vec_clf.predict([review]))
#flat_list = [item for sublist in labels for item in sublist]
import itertools
flat_list2=list(itertools.chain.from_iterable(labels))

dataset = pd.DataFrame({'condi':condi,'labels':flat_list2} ,columns=['condi', 'labels'])
dataset.insert(0, 'company', db['company']) #put company name back

# ###################
# #
# #Let's make columns by searching for keywords!!!!!! So much fun!!!!!!
# #We should also gave it a score based on how many reviews of each type + or - we got
# #Let's build some functions
# #
# ##################

def make_array2(company_name):
    keywordis=['food','free coff','surf','pet-fr','student loan pay','gym','yoga','perk','stress', 'divers','rais','vacat','health', 'women','ice cream','massag' 'benefit', 'balanc', 'discount', 'incent', 'pay', 'sign bonus', 'stock option', 'work home', 'work environ','free lunch']
    a_company=[]
    for key in keywordis:
        a_company.append(column_builder2(key))
    return a_company
def column_builder2(string):
    some_string=dataset[dataset['condi'].str.contains(string)] #looks for string in conditioned data
    some_string.insert(0,'keyword',string)
    return  some_string

temp_table=make_array2(dataset)
for element in temp_table:
    db =pd.concat([element,db], axis=0, ignore_index=False)

#we shall group by company/keyword to add all the sentiments per keyword to get a score
g_b=db.groupby(['company','keyword'])
sumit=g_b.sum('labels')
cluster=sumit.pivot_table('labels', ['company'], 'keyword')

#lets get rid of the NA's, make them 0, because I can + makes sence
cluster=cluster.fillna(0)
cluster=cluster.reset_index()
cluster.to_csv('clusters.csv')


