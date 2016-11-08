import gensim as gm
import pandas as pd
import re
from nltk.corpus import stopwords
data = pd.read_csv("C:\\Users\\Patrick\\Documents\\Reviews.csv")
topic_number = 150
lda = gm.models.ldamodel.LdaModel.load("lda.model")

dictionary = gm.corpora.Dictionary.load("dictionary.dd")
#Assign topics to each review as columns in a new CSV where the topic weights are added as a matrix beside the previous information; 10 columns - > 10 + number of topics 
#Bring it into R for statistical analysis

data = data[['Id', 'Text']]
docs = data[['Text']].values.tolist()
docs = [re.sub(r'[^a-zA-Z0-9\s\:]', '', doc[0]).lower().split() for doc in docs]

#for i in 1:150 if i in lda[doc] then dict(lda[doc])[i] else 0
#[dict(lda[dictionary.doc2bow(docs[x])])[i] if i in dict(lda[dictionary.doc2bow(docs[x])]) else 0 for i in range(151)]

    
def all_topics(doc):
    topics = dict(lda[dictionary.doc2bow(doc)])
    return([topics[i] if i in topics else 0 for i in range(topic_number)])
    

list_df = [all_topics(i) for i in docs]
    
df = pd.DataFrame(list_df)

df.to_csv("topics.csv")
