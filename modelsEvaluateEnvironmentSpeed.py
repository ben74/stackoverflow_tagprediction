#py3 models.py
import os,sklearn.preprocessing;
#os.system('pip install pysftp webptools bs4 lightgbm;');
#os.system('rm -f benLib1.py;wget https://alpow.fr/benLib1.py')
import alpow;from alpow import *
#import importlib;importlib.reload(alpow);#reload
#alpow.sftp=
alpow.demo=0;
alpow.verbose=0
jeuDonnees={}

def load(fn='allVars'):
  fns=fn.split(',')
  for fn in fns:
    fn=fn.strip(', \n')
    if(len(fn)==0):
      continue
    globals().update(benLib1.resume(fn))

def save(exc=[],fn='allVars',include=False,backup=False,ftp=True,cleanup=False,zip=True,authTypes=[str,dict,list,int,np.ndarray,pd.DataFrame,pd.Series]):
  return;
  if(type(exc)==str):#quicksave single var
    excs=exc.split(',')
    for exc in excs:
      exc=exc.strip(', \n')
      if(len(exc)==0):
        continue
      fn=exc
      include=[exc]
      exc=[]
      benLib1.save(globals(),exclusions=exc,fn=fn,include=include,backup=backup,ftp=ftp,cleanup=cleanup,zip=zip,authTypes=False)
    print(excs)
    return 1
  elif exc==[]:
    exc=exclusions;
  benLib1.save(globals(),exclusions=exc,fn=fn,include=include,backup=backup,ftp=ftp,cleanup=cleanup,zip=zip,authTypes=authTypes)

cv_results={}
#Mdf,subject
#les grosses variables
###############}{
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from functools import partial
from sklearn.feature_extraction.text import TfidfVectorizer
# For multiclass classification
from sklearn.multiclass import OneVsRestClassifier

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from lightgbm import LGBMClassifier
from collections import Counter
from itertools import chain
import scipy

import sklearn.multioutput
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB
from skmultilearn.problem_transform import ClassifierChain
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import hamming_loss
from sklearn.cluster import KMeans

import IPython
import multiprocessing
cores = multiprocessing.cpu_count() 

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from functools import partial
import gc
import re
from bs4 import BeautifulSoup
from nltk.tokenize import ToktokTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from string import punctuation

import sklearn.multioutput
from functools import partial

import glob,nltk
from sklearn import preprocessing
from sklearn import decomposition

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px
import numpy as np 
np.random.seed(seed=1983)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings,psutil
def ramUsage():
  process = psutil.Process(os.getpid())
  return round(process.memory_info().rss/1024/1024,2)  # in M

def print_evaluation_scores(y_val, predicted):
    f1_score_micro = partial(f1_score,average="micro")
    return f1_score_micro(y_val,predicted)

    f1_score_macro = partial(f1_score,average="macro")    
    f1_score_weighted = partial(f1_score,average="weighted")
    
    average_precision_score_macro = partial(average_precision_score,average="macro")
    average_precision_score_micro = partial(average_precision_score,average="micro")
    average_precision_score_weighted = partial(average_precision_score,average="weighted")
    
    scores = [accuracy_score,f1_score_macro,f1_score_micro,f1_score_weighted,average_precision_score_macro,
             average_precision_score_micro,average_precision_score_weighted]
    for score in scores:
        print(score,score(y_val,predicted))
        

nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')
tokenizer = nltk.RegexpTokenizer(r'\w+')

pd.set_option('display.max_rows',900)
pd.set_option('display.max_columns',40)
pd.set_option('display.width',1200)#or evaluate js document frame innerWidth .
plt.style.use('fivethirtyeight')
plt.rcParams["figure.figsize"] = (24,12)
plt.rcParams['figure.facecolor'] = 'white'

###

TF_IDF_matrix_train={}
TF_IDF_matrix_test={}
TF_IDF_matrix={}
perplexity={}
featnames={}
scores={}
lda={}
f1s={}
classifiers={}
totScore={}
totScoreTrain={}

#Most frequent keywords associations ==> synonymes, champ lexical
#reduceList(200,list)


def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("_"*180)
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

warnings.filterwarnings('ignore')

def unik(x):
  if type(x)==np.ndarray:
    return set(x)
#if dataframe    
  return x.value_counts()

def vc(x):
  v=x.value_counts()#print(v.sum())
  return v.sum()

def loadData(f,sep=','):
  return pd.read_csv(f,sep=',')
def unikPerCol(df,i):
  return df[df[i]!=0].groupby(i)[i].count().sort_values(ascending=False)
def unikValuesPerDataframe(df,exlude='z_'):
  u={}
  #colonnesStrings=train.select_dtypes(include=['object']).columns
  colonnesStrings=df.columns.values
  for i in colonnesStrings:
    if i.startswith(exlude):
      continue
    suffix=''
    if df.dtypes[i]==object:
      df[i]=df[i].str.lower();#cast All Strings to lowercase AND TRIM ?
    else:
      suffix=' (mean:'+str(df[i].mean())+')'

    u[i]=len(df[i].unique())
    x=df[df[i]!=0].groupby(i)[i].count().sort_values(ascending=False)
    allUnique='';
    if(x.values[0]==1):
      allUnique=' : all unique'

    print('_'*120)
    print(' ?? Different values in '+i+' : '+str(u[i])+suffix+allUnique)
    if(x.values[0]>1):
      print(x.head(5))

def unikValues(x):
  return x.value_counts();

def unikGt1(x):
  _s=x.value_counts();
  gt1=_s[_s.gt(1)];
  return len(gt1)

  #radar stuff
#radar(_df[:1]['columns'.split(',')], 'titre',1)
def radar(_df,title='radar',seuilSuppression=0):
  fn=title+'.png'
  sel=_df.copy()
  #print(sel);print(sel['product_name'].values[0]);print(sel['code'].values[0])
  #print(sel.keys())
  for i in sel.keys():
    #print(i,type(sel[i].values[0]))#<class 'numpy.int64'>
    #print(i);print(sel[i].values[0])
    if type(sel[i].values[0])=='object':
      del(sel[i].values[0]);
    elif(sel[i].values[0]<seuilSuppression):
      print('del',i)
      del(sel[i]);
      #print('del:'+i)
#Exception: Data must be 1-dimensional
  #radardf = sel;#pd.DataFrame(dict(  r=sel.values[0], theta=sel.columns))
  radardf = pd.DataFrame(dict(columns=sel.columns,values=sel.values[0]))
  #print(radardf)
#color="strength", symbol="strength", size="frequency",color_discrete_sequence=px.colors.sequential.Plasma_r  
  #scatter_polar
  fig = px.scatter_polar(radardf,r='values',theta='columns',title=title)
  #fig = px.line_polar(radardf,r='values',theta='columns', line_close=True,title=title)
  fig.update_traces(fill='toself')
  #f.savefig(fn,bbox_inches='tight');webp(fn);
  fig.show()

#radar(pd.DataFrame({'yo':40,'ya':20,'yi':70,'yu':90,'ye':25}, index=[0]),'titre',0) 
#radar(x,'titre',0)  
def kde(var,fn,quantilex=1,ax=False):
  save=0
  if ax==False:
    save=1
    f,ax=plt.subplots(1,1)
    f.patch.set_facecolor('white');
  ax.set_title(fn)
  ax.set_xlim(0,var.quantile(quantilex))#Avec les limites adaptées à chaque type de données  
  var.plot(kind='kde',ax=ax).yscale='log';
  if save==1:
    f.savefig(fn,bbox_inches='tight');webp(fn);plt.close()

print('use quickload then')


####}{
    #load('sqldf')
#Design Patterns Bullshit & Dette Technique
from nltk import sent_tokenize, word_tokenize
from nltk.tokenize import ToktokTokenizer

#copie de travail

import nltk
nltk.download('wordnet')
nltk.download('stopwords')

#stopwords=nltk.corpus.stopwords.words('english')
#tokenizer = nltk.RegexpTokenizer(r'\w+')
lemma=nltk.WordNetLemmatizer()
token=ToktokTokenizer()
#stop_words = set(stopwords.words("english"))
stop_words=nltk.corpus.stopwords.words('english')

def clean_text(text):
  #remove punctuation , ? . /
    return text.lower()#.decode('utf-8').lower())


#Stemmer is Faster .. Lemmitize prend en compte le sens des phrases
def lemitizeWords(text):
    words=token.tokenize(text)
    listLemma=[]
    for w in words:
        x=lemma.lemmatize(w, pos="v")
        listLemma.append(x)
    return ' '.join(map(str, listLemma))

def stopWordsRemove(text):
    words=token.tokenize(text)
    filtered = [w for w in words if not w in stop_words]
    return ' '.join(map(str, filtered))

#conserver #.
punct = '!"$%&\'()*+,/:;<=>?@[\\]^_`{|}~'
regex = re.compile('[%s]' % re.escape(punct))

def clean_punct(text): 
    words=token.tokenize(text)
    punctuation_filtered = []    
    remove_punctuation = str.maketrans(' ', ' ', punct)
    for w in words:
        if w in tags_features:
            punctuation_filtered.append(w)
        else:
            punctuation_filtered.append(regex.sub('', w))
  
    filtered_list = strip_list_noempty(punctuation_filtered)
        
    return ' '.join(map(str, filtered_list))    

def stripTags(x):
  return re.sub(r"[<> ]+",' ', x).lower().strip();

def strip_list_noempty(mylist):
    newlist = (item.strip() if hasattr(item, 'strip') else item for item in mylist)
    return [item for item in newlist if item != '']

#}}}}
load('df')
titleWeight=4
nbRows=10000 #one empty y_train vector, why ???
#nb de mots considérés
DICT_SIZE = 5000
k2=str(nbRows)+'_'+str(DICT_SIZE)+'_'+str(titleWeight)
print('segmentation:',k2)
jeuDonnees[k2]={}


from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from functools import partial
from sklearn.feature_extraction.text import TfidfVectorizer
# For multiclass classification
from sklearn.multiclass import OneVsRestClassifier

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from lightgbm import LGBMClassifier
from collections import Counter
from itertools import chain
import scipy

def my_bag_of_words(text, words_to_index, dict_size):
  result_vector = np.zeros(dict_size)
  keys= [words_to_index[i] for i in text.split(" ") if i in words_to_index.keys()]
  result_vector[keys]=1
  return result_vector

def tfidf_features(X_train, X_test):
  tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2),max_df=0.9,min_df=5,analyzer = 'word',token_pattern=r"(?u)\S\S+" )
  tfidf_vectorizer.fit(np.concatenate((X_train, X_test)))
  X_train = tfidf_vectorizer.transform(X_train)
  X_test = tfidf_vectorizer.transform(X_test)
  return X_train, X_test, tfidf_vectorizer.vocabulary_

#ajout du body
tags=df['tags'][:nbRows].values
tags=[','.join(text) for text in tags]
#body df+' '+df['body']

#£:rajouter un champ au sein du dataframe => (103,426,246) => soit la plus fidèle possible jeu échantilloné dispose même répartion python,javascript,ajax sur les top 20 tags
#£:Stratification : ScikitML Stratification
#repeat n fois df['title'][:nbRows] +
X_train, X_test, y_train, y_test = train_test_split( df['title'][:nbRows] * titleWeight + df['body'][:nbRows] ,tags, test_size=0.2, random_state=42)
#X_train, X_test, y_train, y_test = train_test_split(df['title'][:nbRows].values, tags, test_size=0.2, random_state=42)
# Dictionary of all tags from train corpus with their counts.
tags_counts = Counter(chain.from_iterable([i.split(',') for i in y_train]))#

# Dictionary of all words from train corpus with their counts.
words_counts = Counter(chain.from_iterable([i.split(" ") for i in X_train]))

top_tags = sorted(tags_counts.items(), key=lambda x: x[1], reverse=True)[:50]
top_words = sorted(words_counts.items(), key=lambda x: x[1], reverse=True)[:50]
#ALL_WORDS,WORDS_TO_INDEX,INDEX_TO_WORDS
print(f"Top 50 most popular tags are: {','.join(tag for tag, _ in top_tags)}")
print(f"Top 50 most popular words are: {','.join(tag for tag, _ in top_words)}")

WORDS_TO_INDEX = {j[0]:i for i,j in enumerate(sorted(words_counts.items(), key=lambda x: x[1], reverse=True)[:DICT_SIZE])}
INDEX_TO_WORDS = {i:j[0] for i,j in enumerate(sorted(words_counts.items(), key=lambda x: x[1], reverse=True)[:DICT_SIZE])}
ALL_WORDS = list(WORDS_TO_INDEX.keys())

if'Bag of words':
  X_train_mybag = scipy.sparse.vstack([scipy.sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_train])
  X_test_mybag = scipy.sparse.vstack([scipy.sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_test])  
  print('X_train shape ', X_train_mybag.shape)
  print('X_test shape ', X_test_mybag.shape)

if'IDF':
  X_train_tfidf, X_test_tfidf, tfidf_vocab = tfidf_features(X_train,X_test)
  tfidf_reversed_vocab = {i:word for word,i in tfidf_vocab.items()}


# transform y_train to dictionary
#load('y_train,y_test')
y_train = [set(i.split(',')) for i in y_train]
y_test = [set(i.split(',')) for i in y_test]

mlbTrain = MultiLabelBinarizer()
y_train = mlbTrain.fit_transform(y_train)
y_test2 = mlbTrain.transform(y_test)#472 ok
print('Y_test shape ', np.array(y_test2).shape)
print('Y_train shape ', np.array(y_train).shape)

#print(y_test2);assert(False)





#####}{Adaptes le toujours au plus simple

print('_'*180)

jeuDonnees[k2]={'myBag':[X_train_mybag,X_test_mybag],'tfIdf':[X_train_tfidf,X_test_tfidf]}
variables='y_train,y_test,y_test2,mlbTrain,tfidf_vocab,tfidf_reversed_vocab,ALL_WORDS,WORDS_TO_INDEX,INDEX_TO_WORDS'.split(',')
#for i in variables:
  #jeuDonnees[k2][i]=globals()[i]  
  #del(globals()[i])
#}}}}

#load('jeuDonnees');
#k2='10000_5000_4';extract(jeuDonnees[k2])#
load('y_test2,y_train')
times={}
grids={
  'rdf':{
    'estimator__random_state':[42],
    'estimator__n_estimators':[10,50,100],
    'estimator__n_jobs':[-1],
  },
  'svm':{
    'estimator__random_state':[42],
    'estimator__C':[1,5,10],
    'estimator__alpha':[1e-3],
  } 
}

models={
  'lr':LogisticRegression(C=3, penalty='l1', dual=False, n_jobs=-1,random_state=42),
  'svm':LinearSVC(C=1, penalty='l1', dual=False, loss='squared_hinge',random_state=42),
  'nbayes':MultinomialNB(alpha=1.0), #Mauvais résultats !,
  'dummy':DummyClassifier(random_state=42),
  'sgd':SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42,max_iter=5, tol=None, n_jobs=-1),
  'rdf':RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
  #,'lda':LinearDiscriminantAnalysis(solver='svd')
}
#models=['lda'] #prend vraiment bcp trop de temps
#cchain: takes too much time & LabelPowerSet ScikitML.LP 
groupers={'mloc':sklearn.multioutput.MultiOutputClassifier,'oneVsRest':OneVsRestClassifier}#,'cchain':ClassifierChain

for j in list(models.keys()):  

  for k in groupers:
    grouper=groupers[k]    
    for l in 'tfIdf|myBag'.split('|'):
      print('_'*180)
      k3=j+'_'+k+'_'+l
      print(j,'-',k,'-',l)
      #importance de conserver k2 référentiel segmentation jeu de données
      x='model_'+l+'_'+j+'_'+k+'_'+k2
      if False:
        if(os.path.exists(x+'.pickle')):
          print('allready exists : ',x)
          continue;
        
        getFile(x)#tentative récupération distante
        if(os.path.exists(x+'.tgz')):
          print('allready exists : ',x)
          continue;

      y = grouper(models[j])
      
      xtrain,xtest=jeuDonnees[k2][l]#inputTrainData

      grid={}
      if(j in grids.keys()):
        grid=grids[j]

      globals()[x] = GridSearchCV(estimator=y,param_grid={},n_jobs=-1,cv=5,scoring='f1_micro')
      a=time()
#ValueError: This solver needs samples of at least 2 classes in the data, but the data contains only one class: 0      
      globals()[x].fit(xtrain, y_train)
      b=time()-a
      times[x]=b
      cv_results[k3] = globals()[x].cv_results_
      print("Best parameters from gridsearch: {}".format(globals()[x].best_params_))
      print("CV score=%0.3f" % globals()[x].best_score_)

      ypred=globals()[x].predict(xtrain)
      totScoreTrain[x+'_train']=print_evaluation_scores(y_train, ypred)

      ypred=globals()[x].predict(xtest)
      totScoreTrain[x+'_test']= print_evaluation_scores(y_test2, ypred)
      print(l,' ram:',ramUsage(),',time:',int(b),' secs, test score:',totScoreTrain[x+'_test'],', train score:',totScoreTrain[x+'_train']);
      
      del(globals()[x]);gc.collect()
print(times)      