import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# This function plots a confusion matrix 
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))  
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# This function plots a confusion matrix without a colorbar to not have a colorbar at each subplot when printing cross validation
def plot_confusion_matrix_without_colorbar (cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    classes = ['1', '2', '3', '4', '5', '6']

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    #plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# This function reads the corpus and returns the document containing the review's text together with either the sentiment (use_sentiment = True) or topics (use_sentiment = False)
def read_corpus(corpus_file, use_sentiment):
    documents = []
    labels = []
    with open(corpus_file, encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            documents.append(tokens[3:])

            if use_sentiment:
                # 2-class problem: positive vs negative
                labels.append(tokens[1])
            else:
                # 6-class problem: books, camera, dvd, health, music, software
                labels.append(tokens[0])

    return documents, labels
    
# a dummy function that just returns its input
def identity(x):
    return x

# Reading the dataset with X containing the review's text and Y as labels which could be sentiment (use_sentiment = True) or topics (use_sentiment = False)
# Then the dataset is split in a train (75%) and test (25%) set  
X, Y = read_corpus('trainset.txt', use_sentiment=False)

#prints the frequency count of the labels
print("The term frequencies are")
d = {x:Y.count(x) for x in Y}
print(d)

#split_point = int(0.9*len(X))
split_point = int(0.75*len(X))
Xtrain = X[:split_point]
Ytrain = Y[:split_point]
Xtest = X[split_point:]
Ytest = Y[split_point:]

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

# combine the vectorizer with a Naive Bayes classifier
classifier = Pipeline( [('vec', vec),
                        ('cls', MultinomialNB())] )


# The classifier is traassigned to them (why are posteriors different from priors?).ined on the Xtrain and Ytrain dataset
classifier.fit(Xtrain, Ytrain)

# The classfier predicts the Ygeuss based on the Xtest after training
Yguess = classifier.predict(Xtest)


#shows the precision, recall and f-score per class
from sklearn.metrics import classification_report
#target_names = ['class 1', 'class 2']
target_names = ['class 1', 'class 2', 'class 3', 'class 4', 'class 5', 'class 6']
print(classification_report(Ytest, Yguess, target_names=target_names))
 
# The accuracy is calculated by a comparison of the actual labels and the guessed labels
print("accuracy")
print(accuracy_score(Ytest, Yguess))

#The accuracy score per class can be obtained by the normalized confusion matrix. Diagonal values are accuracy per class
cnf_matrix = confusion_matrix(Ytest, Yguess)
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=set(Ytest), normalize=True,
                      title='Normalized confusion matrix')  

# Calculate precision micro and macro
from sklearn.metrics import precision_score
print("precision")
precision_macro = precision_score(Ytest, Yguess, average='macro') 
print(precision_macro)
precision_micro = precision_score(Ytest, Yguess, average='micro') 
print(precision_micro)

# Calculate recall micro and macro
from sklearn.metrics import recall_score
print("recall")
recall_macro = recall_score(Ytest, Yguess, average='macro')
print(recall_macro)
recall_micro = recall_score(Ytest, Yguess, average='micro')
print(recall_micro)

# Calculate f1_score micro and macro
from sklearn.metrics import f1_score
print("f-score")
f_score_macro = f1_score(Ytest, Yguess, average='macro')  
print(f_score_macro)
f_score_micro = f1_score(Ytest, Yguess, average='micro')  
print(f_score_micro)


# Compute confusion matrix
cnf_matrix = confusion_matrix(Ytest, Yguess)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=set(Ytest),
                      title='Confusion matrix, without normalization')

plt.show()


# cross validation
from sklearn.model_selection import KFold
kf = KFold(n_splits=5)
kf.get_n_splits(X)

X = np.array(X)
Y = np.array(Y)


plt.figure()

total_count = np.zeros((6,6), dtype=int)
y_pred = []
fold = 1

for train_index, test_index in kf.split(X):
  print("TRAIN:", train_index, "TEST:", test_index)
  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = Y[train_index], Y[test_index]

  # train the classifier on X
  classifier.fit(X_train, y_train)

  # The classfier predicts the Ygeuss based on the Xtest after training
  y_guess = classifier.predict(X_test)
  y_pred.extend(y_guess)
  # Compute confusion matrix
  cnf_matrix = confusion_matrix(y_test, y_guess)
  np.set_printoptions(precision=2)

  plt.subplot(3,2,fold)
  # Plot non-normalized confusion matrix
  plot_confusion_matrix_without_colorbar(cnf_matrix, classes=set(Ytest),
                      title='')
  fold = fold + 1
  total_count = total_count + cnf_matrix

plt.subplot(3,2,fold)
plot_confusion_matrix_without_colorbar(total_count, classes=set(Ytest),
                      title='')
plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
#plt.tight_layout()
cax = plt.axes([0.85, 0.1, 0.075, 0.8])
plt.colorbar(cax=cax)
plt.show()

print(classification_report(Y, y_pred))
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(total_count, classes=set(Ytest), normalize=True,
                      title='Normalized total confusion matrix')  
plt.show()


# The code below shows the probabilities for the test examples belonging to a class 
classifier.fit(Xtrain, Ytrain)
print("Probabilities")
print(classifier.predict_proba(Xtest))