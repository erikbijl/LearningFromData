import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

import sys


def read_corpus(corpus_file, use_sentiment):
    documents = []
    labels = []
    with open(corpus_file, encoding='utf-8') as f:
        for line in f:
            
            tokens = line.strip().split()
            
            #replace all numbers to the string 'number' 
            #new_tokens = []
            #for token in tokens: 
            #    if token.isdigit():
            #        new_tokens.append('digit')
            #    else:
            #        new_tokens.append(token)
            #tokens = new_tokens
                    
            # remove stopwords
            from nltk.corpus import stopwords
            stop = stopwords.words('english')
            tokens = [token for token in tokens if token not in stop]
            
            # porter stemming
            from nltk.stem.porter import PorterStemmer
            st = PorterStemmer()
            tokens = [st.stem(word) for word in tokens]
            
            documents.append(tokens[3:])

            if use_sentiment:
                # 2-class problem: positive vs negative
                labels.append( tokens[1] )
            else:
                # 6-class problem: books, camera, dvd, health, music, software
                labels.append( tokens[0] )

    return documents, labels
    
# a dummy function that just returns its input
def identity(x):
    return x

# read the arguments
train_set = sys.argv[1]
test_set = sys.argv[2]

# reads the corpus 
print("reading train set")
print(train_set)
X_train, Y_train = read_corpus(train_set, use_sentiment=False)
print("reading test set")
print(test_set)
X_test, Y_test = read_corpus(test_set, use_sentiment=False)

# let's use the TF-IDF vectorizer
tfidf = True

# we use a dummy function as tokenizer and preprocessor,
# since the texts are already preprocessed and tokenized.
if tfidf:
    vec = TfidfVectorizer(preprocessor = identity,
                          tokenizer = identity)
else:
    vec = CountVectorizer(preprocessor = identity,
                          tokenizer = identity)


print("perform naive bayes classification with a grid search")
# NAIVE BAYES
params = {'cls__alpha': np.arange(0.50, 0.60, 0.01)}

# combine the vectorizer with a Naive Bayes classifier
classifier = Pipeline( [('vec', vec),
                        ('cls', MultinomialNB())] )

GS = GridSearchCV(classifier, params, cv=5, scoring='f1_micro')
GS.fit(X_train, Y_train)

print("Found best value for alpha:")
print(GS.best_params_)
print(GS.best_score_)


# combine the vectorizer with a Naive Bayes classifier with best value
classifier = Pipeline( [('vec', vec),
                        ('cls', MultinomialNB(alpha=GS.best_params_['cls__alpha']))] )

print("fit the train set")
classifier.fit(X_train, Y_train) 
print("classification report:")
print(classification_report(classifier.predict(X_test), Y_test))