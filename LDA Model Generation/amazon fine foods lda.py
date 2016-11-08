import gensim as gm
import pandas as pd
import re
from nltk.corpus import stopwords
stopwords = stopwords.words('english')
data = pd.read_csv("C:\\Users\\Patrick\\Documents\\Reviews.csv")
data = data[['Id', 'Text']]
docs = data[['Text']].values.tolist()
#re.sub(r'[^a-zA-Z0-9\s\:]', '', doc)
#docs = [doc[0].lower().split() for doc in docs]
docs = [re.sub(r'[^a-zA-Z0-9\s\:]', '', doc[0]).lower().split() for doc in docs]


dictionary = gm.corpora.Dictionary(docs)
stop_list_ids = [dictionary.token2id[i] for i in stopwords if i in dictionary.token2id]
len_2_list = [dictionary.token2id[i] for i in dictionary.token2id if len(i) <= 2]
bad_ids =  len_2_list + stop_list_ids
dictionary.filter_tokens(bad_ids=bad_ids)
dictionary.filter_extremes(no_below = 20, no_above = 0.75, keep_n = None)

corpus = [dictionary.doc2bow(doc) for doc in docs]

