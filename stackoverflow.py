#}{handles the json payload served by the server
#cd /home/python;py3 stackoverflow.py
import os;os.system('pip install pysftp;pip install scikit-multilearn;pip install webptools;pip install pyLDAvis;rm -f alpow.py;wget https://alpow.fr/alpow.py');import alpow;from alpow import *
basedir='http://1.x24.fr/a/jupyter/stackXchange/'
#alpow.sftp
alpow.demo=0;
alpow.verbose=0
jeuDonnees={}
times={}
predictProbas={}
#on resume check if exists
ftplist=ftpls();

def load(fn='allVars',onlyIfNotSet=1):
  fns=fn.split(',')
  for fn in fns:
    fn=fn.strip(', \n')
    ok=1
    if(len(fn)==0):
      continue
    if(onlyIfNotSet):
      if fn in globals().keys():
  #override empty lists, dict, dataframe and items      
        if type(globals()[fn])==type:
          continue;
        elif type(globals()[fn])==pd.DataFrame :
          if globals()[fn].shape[0]>0:            
            continue
        elif(type(globals()[fn])==dict):
          if(len(globals()[fn])>0):
            continue
        elif(type(globals()[fn])==list):
          if(len(globals()[fn])>0):
            continue
  #si déjà définie, passer au prochain     
        elif(globals()[fn]):
          continue
    globals().update(alpow.resume(fn))
  #endfor fn
  return;


def extract(x):
  liste=list(x.keys())
  for i in liste:
    globals()[i]=x[i]
  print('extracted : ',','.join(liste))

#jeuDonnees=compact('y_test,')
def compact(variables):
  x={}
  for i in variables.split(','):
    x[i]=globals()[i]    
  print('compacted : ',variables)
  return x

def loadIfNotSet(x):
  if x not in globals().keys():
    load(x)

def save(exc=[],fn='allVars',include=False,backup=False,ftp=True,cleanup=False,zip=True,authTypes=[str,dict,list,int,np.ndarray,pd.DataFrame,pd.Series]):
  global ftplist;
  if(type(exc)==str):#quicksave single var
    excs=exc.split(',')
    for exc in excs:
      exc=exc.strip(', \n')
      if(len(exc)==0):
        continue
      fn=exc
      include=[exc]
      exc=[]
      alpow.save(globals(),exclusions=exc,fn=fn,include=include,backup=backup,ftp=ftp,cleanup=cleanup,zip=zip,authTypes=False)
    print(excs)
    return 1
  elif exc==[]:
    exc=exclusions;
  alpow.save(globals(),exclusions=exc,fn=fn,include=include,backup=backup,ftp=ftp,cleanup=cleanup,zip=zip,authTypes=authTypes)
  ftplist=ftpls();

cv_results={}
#Mdf,subject
#les grosses variables
###############}{
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
#%matplotlib inline 

import warnings,psutil
def ramUsage():
  process = psutil.Process(os.getpid())
  return round(process.memory_info().rss/1024/1024,2)  # in M

def print_evaluation_scores(y_val, predicted):###!
  return {'jacard':round(avg_jacard(y_val, predicted),4),'hamming':round(sklearn.metrics.hamming_loss(predicted, y_val),4),'f1':round(sklearn.metrics.f1_score(y_val, predicted, average='micro'),4)};
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
  tid={}
  for topic_idx, topic in enumerate(model.components_):
    #print("_"*180)
    #print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
    tid[topic_idx]=' '.join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]])
  return tid
warnings.filterwarnings('ignore')

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

num_topics=40

def get_doc_topic_dist(model,corpus1,kwords=False):
  lda_keys=[]
  top_dist =[]
  for d in corpus1:
    tmp = {i:0 for i in range(num_topics)}
    tmp.update(dict(model[d]))
    vals = list(OrderedDict(tmp).values())
    top_dist += [pylab.array(vals)]
    if kwords:
      lda_keys += [pylab.array(vals).argmax()]
  return top_dist,lda_keys

def proba(_mdl,x,minus=1):
  global probasPerClass,top2,QuestionToTags
  rows=x.shape[0]
  probaTagId2={}
  probasPerClass=_mdl.predict_proba(x)#1600,2 => pprobas[0].shape #1600,2 pprobas.shape
  i=0
  simple=0
  nbTags=len(probasPerClass);

  if(type(probasPerClass[0][0])==np.float64):
    probaTagId2=probasPerClass
    nbTags=len(probasPerClass[0]);
    simple=1
  
  #print('nbtags:',nbTags)
  if not simple:
    while(i<nbTags-minus):#1370
      if(len(probasPerClass[i][0])>1):
        probaTagId2[i]=probasPerClass[i][:,1]
      else:
        probaTagId2[i]=[0]*rows;
      i+=1

  probaIdPerQeustion=pd.DataFrame(probaTagId2)
#¤todo:si seuil > tant
  top2=probaIdPerQeustion.apply(lambda x: pd.Series(np.concatenate([x.nlargest(2).index.values])), axis=1)  
  QuestionToTags=top2.T.to_dict()
  emptyMatrix=[0]*nbTags

  ypredictions=[]  
  for i in list(QuestionToTags.keys()):
    y=emptyMatrix
    for j in list(QuestionToTags[i].values()):
      y[j]=1
    ypredictions+=[y]

  return ypredictions  

#1 mot vaut un dans une dimension, plusieur usages
def my_bag_of_words(text, words_to_index, dict_size):
  result_vector = np.zeros(dict_size)
  keys= [words_to_index[i] for i in text.split(" ") if i in words_to_index.keys()]
  result_vector[keys]=1
  return result_vector

from keras import backend as K
def recall_m(y_true, y_pred):
  true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
  recall = true_positives / (possible_positives + K.epsilon())
  return recall

def precision_m(y_true, y_pred):
  true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
  precision = true_positives / (predicted_positives + K.epsilon())
  return precision

def f1_m(y_true, y_pred):
  precision = precision_m(y_true, y_pred)
  recall = recall_m(y_true, y_pred)
  return 2*((precision*recall)/(precision+recall+K.epsilon()))
  
def avg_jacard(y_true,y_pred):
    jacard = np.minimum(y_true,y_pred).sum(axis=1) / np.maximum(y_true,y_pred).sum(axis=1)
    return jacard.mean()#;*100

#maxdf=0.0080 #95% IDF -> 20% is recommanded
rds=42 
def split(df,nb,DICT_SIZE = 1000,encode=True,targetVariance=.9,pca=True,test_size=.2,mindf=5.5e-05 ,maxdf=0.0080,maxfeatures=1000,titleWeight=4):
  from collections import Counter
  from itertools import chain
  from sklearn.utils import class_weight

  dftrain,dftest=train_test_split(df[:nb],test_size=test_size,random_state=rds)  

  y=0
  yoriginal = df['tags'][:nb]

  y2Occurences=[]
  for i in yoriginal:#1 to index
    y2Occurences+=i
  y2Occurences  

  class_weight = class_weight.compute_class_weight('balanced',np.unique(y2Occurences),y2Occurences)

  inputx= df['title'][:nb]*titleWeight+ ' '+df['body'][:nb]
  words_counts = Counter(chain.from_iterable([i.split(" ") for i in inputx]))
  WORDS_TO_INDEX = {j[0]:i for i,j in enumerate(sorted(words_counts.items(), key=lambda x: x[1], reverse=True)[:DICT_SIZE])}
  INDEX_TO_WORDS = {i:j[0] for i,j in enumerate(sorted(words_counts.items(), key=lambda x: x[1], reverse=True)[:DICT_SIZE])}
  ALL_WORDS = list(WORDS_TO_INDEX.keys())

  MBW=[]
  for text in inputx:
    MBW+=[my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)]
  if pca:
    pca2=sklearn.decomposition.PCA(n_components=targetVariance,random_state=rds)
    MBW=pca2.fit_transform(MBW)
    sumVariance=round(list(pca2.explained_variance_ratio_.cumsum()).pop(),2)
    p('NbDimensions:',MBW.shape[1],', sum variance:',sumVariance)
  #MBW=pca2.transform(MBW)
  mlbTrain=0
  if(encode):
    mlbTrain = MultiLabelBinarizer()
    y = mlbTrain.fit_transform(yoriginal)  

  if'returns the TFIDF output as well':
    TFvectorizer = TfidfVectorizer(analyzer='word',min_df=mindf,max_df=maxdf,strip_accents = None,encoding = 'utf-8', preprocessor=None,max_features=maxfeatures,token_pattern=r"(?u)\S\S+")
    a=time()
    TFvectorizer.fit(inputx)
    b=time();c=b-a;p('tfidf fits in ',c,'secs')
    TFIDF=TFvectorizer.transform(inputx)

  X_train, X_test, y_train, y_test = train_test_split(MBW , y, test_size=test_size, random_state=rds)#,stratify=y
  TFIDF_train, TFIDF_test = train_test_split(TFIDF, test_size=test_size, random_state=rds)#,stratify=y
  return X_train,y_train,X_test,y_test,mlbTrain,MBW,y,yoriginal,class_weight,dftrain,dftest,TFIDF_train,TFIDF_test


def trainModels():
  global scores;
  alpow.p=p=nf
  ftplist=ftpls()
  for i in list(models.keys()):
    echo('_'*180);
  #bag or tfidf
    for j in list(train2predictions.keys()):
      train2prediction=train2predictions[j]
      x='model_'+i+'_'+j+'_'+k
      echo('_'*180);
      echo(x)
      if(x+'.tgz' in ftplist):
        load(x)    
      if(x not in globals().keys()):    
        xt1=train2prediction[0][0]
        globals()[x]=models[i][0]
        if((len(models[i])>1) & (j in scalers.keys())):
          xt1=scalers[j].transform(xt1)
  
        globals()[x].fit(xt1, y_train)
        save(x)
      mdl=globals()[x]  
  #train,test
      if(i+'_'+k not in scores.keys()):
        scores[i+'_'+k]={}
      for train2predictio in train2prediction:
        echo('_'*100)
        echo(train2predictio[2])    
        if((len(models[i])>1) & (j in scalers.keys())):
          train2predictio[0]=scalers[j].transform(train2predictio[0])    
        simple=mdl.predict(train2predictio[0])
        scores[i+'_'+k][train2predictio[2]]=print_evaluation_scores(simple, train2predictio[1])
        echo('simple:',print_evaluation_scores(simple, train2predictio[1]))
        if hasattr(mdl, 'predict_proba'):
          scores[i+'_'+k][train2predictio[2]+'_predict_proba_']=print_evaluation_scores(proba(mdl,train2predictio[0]), train2predictio[1])   
          pb=proba(mdl,train2predictio[0]);
          echo('predict_proba:',print_evaluation_scores(pb, train2predictio[1]))  
      del(globals()[x],mdl)
  alpow.p=p=print

from collections import Counter
from itertools import chain
from sklearn.utils import class_weight
from skmultilearn.model_selection import iterative_train_test_split
from skmultilearn.model_selection.measures import get_combination_wise_output_matrix


def IterativeSplit(df,nb,DICT_SIZE = 1000,encode=True,targetVariance=.9,pca=True,test_size=.2,mindf=5.5e-05 ,maxdf=0.0080,maxfeatures=1000,titleWeight=4):
  MBW=[] 
  yoriginal=df[:nb]['tags']
  mlbTrain=0
  if encode:
    mlbTrain = MultiLabelBinarizer()
    y = mlbTrain.fit_transform(yoriginal) 
  
  #X_train, y_train, X_test, y_test = iterative_train_test_split(list(df[:nb].index),y,test_size=test_size)
  inputx= df['title'][:nb]*titleWeight+ ' '+df['body'][:nb]
  words_counts = Counter(chain.from_iterable([i.split(" ") for i in inputx]))
  WORDS_TO_INDEX = {j[0]:i for i,j in enumerate(sorted(words_counts.items(), key=lambda x: x[1], reverse=True)[:DICT_SIZE])}
  INDEX_TO_WORDS = {i:j[0] for i,j in enumerate(sorted(words_counts.items(), key=lambda x: x[1], reverse=True)[:DICT_SIZE])}
  ALL_WORDS = list(WORDS_TO_INDEX.keys())

  for text in inputx:
    MBW+=[my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)]
  if pca:
    pca2=sklearn.decomposition.PCA(n_components=targetVariance,random_state=rds)
    MBW=pca2.fit_transform(MBW)
    sumVariance=round(list(pca2.explained_variance_ratio_.cumsum()).pop(),2)
    p('NbDimensions:',MBW.shape[1],', sum variance:',sumVariance)
  _index=df[:nb].index
  #Get Tags not in Train

  X_train, y_train, X_test, y_test = iterative_train_test_split(MBW,y,test_size=test_size)#,random_state=rds  
  return X_train, y_train, X_test, y_test, MBW

  
  

def itSplitDfIndices(df,nb,encode=True,test_size=.2):
  from sklearn.utils import class_weight
  y=df[:nb]['tags']
  y2Occurences=[]
  for i in yoriginal:#1 to index
    y2Occurences+=i
  y2Occurences  
  class_weight = class_weight.compute_class_weight('balanced',np.unique(y2Occurences),y2Occurences)

  mlbTrain=0
  if encode:
    mlbTrain = MultiLabelBinarizer()
    y = mlbTrain.fit_transform(y) 

  MBW=[];I_train=[];I_test=[];
  x=list(df[:nb].index)
  for i in x:
    MBW+=[[i]]
  X_train, y_train, X_test, y_test = iterative_train_test_split(np.array(MBW),y,test_size=test_size)
  
  for i in X_train:
    I_train+=[i[0]]
  for i in X_test:
    I_test+=[i[0]]

  dfTrain=df[df.index.isin(I_train)]
  dfTest=df[df.index.isin(I_test)]
  tagsInTest=dfTest['tags'].sum()
  tagsInTest=unik(tagsInTest)
  tagsInTrain=dfTrain['tags'].sum()  
  tagsInTrain=unik(tagsInTrain)
  diff1=diff(tagsInTest,tagsInTrain)
  echo('iterative train/test split : tagsintest not in train:',len(diff1),' // tagsintest:',len(tagsInTest),' // tagsintrain:',len(tagsInTrain),' //=> ',round((len(diff1)*100)/(len(tagsInTrain)+len(diff1)),2),'%')

  return I_train, y_train, I_test, y_test, y, dfTrain, dfTest, class_weight, mlbTrain#, diff1, tagsInTest, tagsInTrain

#sur le dataframe
def bagIdf(dfTrain,dfTest,DICT_SIZE=1000,targetVariance=.9,pca=True,test_size=.2,mindf=5.5e-05 ,maxdf=0.0080,maxfeatures=1000,titleWeight=4):  
  merged = pd.concat([dfTrain,dfTest])
  #puis split sur mêmes proportions
  inputx= merged['title'][:nb]*titleWeight+ ' '+merged['body'][:nb]
  words_counts = Counter(chain.from_iterable([i.split(" ") for i in inputx]))
  WORDS_TO_INDEX = {j[0]:i for i,j in enumerate(sorted(words_counts.items(), key=lambda x: x[1], reverse=True)[:DICT_SIZE])}
  INDEX_TO_WORDS = {i:j[0] for i,j in enumerate(sorted(words_counts.items(), key=lambda x: x[1], reverse=True)[:DICT_SIZE])}
  ALL_WORDS = list(WORDS_TO_INDEX.keys())
  MBW=[]
  for text in inputx:
    MBW+=[my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)]
  if pca:
    pca2=sklearn.decomposition.PCA(n_components=targetVariance,random_state=rds)
    MBW=pca2.fit_transform(MBW)
    sumVariance=round(list(pca2.explained_variance_ratio_.cumsum()).pop(),2)
    p('NbDimensions:',MBW.shape[1],', sum variance:',sumVariance)
  #MBW=pca2.transform(MBW)

  if'returns the TFIDF output as well':
    TFvectorizer = TfidfVectorizer(analyzer='word',min_df=mindf,max_df=maxdf,strip_accents = None,encoding = 'utf-8', preprocessor=None,max_features=maxfeatures,token_pattern=r"(?u)\S\S+")
    a=time()
    TFvectorizer.fit(inputx)
    b=time();c=b-a;p('tfidf fits in ',c,'secs')
    TFIDF=TFvectorizer.transform(inputx)
  return MBW,TFIDF

def getBagIdfTrainTest(df,nb,DICT_SIZE=1000):
  I_train, y_train, I_test, y_test, y, dfTrain, dfTest, class_weight, mlbTrain = itSplitDfIndices(df,nb)
  MBW,TFIDF = bagIdf(dfTrain,dfTest,DICT_SIZE)
  bag_train, bag_test = MBW[:dfTrain.shape[0]], MBW[dfTrain.shape[0]:]
  TFIDF_test, TFIDF_train = TFIDF[:dfTrain.shape[0]], TFIDF[dfTrain.shape[0]:]
  return bag_train, bag_test,TFIDF_test, TFIDF_train,y_train,y_test,y, dfTrain, dfTest, class_weight, mlbTrain

def trainModels():
  global scores;
  alpow.p=p=nf
  ftplist=ftpls()
  for i in list(models.keys()):
    echo('_'*180);
  #bag or tfidf
    for j in list(train2predictions.keys()):
      train2prediction=train2predictions[j]
      x='model_'+i+'_'+j+'_'+k
      echo('_'*180);
      echo(x)
      if(x+'.tgz' in ftplist):
        load(x)    
      if(x not in globals().keys()):    
        xt1=train2prediction[0][0]
        globals()[x]=models[i][0]
        if((len(models[i])>1) & (j in scalers.keys())):
          xt1=scalers[j].transform(xt1)
  #ValueError: Found input variables with inconsistent numbers of samples: [400, 1600]        
        globals()[x].fit(xt1, y_train)
        save(x)
      mdl=globals()[x]  
  #train,test
      if(i+'_'+k not in scores.keys()):
        scores[i+'_'+k]={}
      for train2predictio in train2prediction:
        echo('_'*100)
        echo(train2predictio[2])    
        if((len(models[i])>1) & (j in scalers.keys())):
          train2predictio[0]=scalers[j].transform(train2predictio[0])    
        simple=mdl.predict(train2predictio[0])
        scores[i+'_'+k][train2predictio[2]]=print_evaluation_scores(simple, train2predictio[1])
        echo('simple:',print_evaluation_scores(simple, train2predictio[1]))
        if hasattr(mdl, 'predict_proba'):
          scores[i+'_'+k][train2predictio[2]+'_predict_proba_']=print_evaluation_scores(proba(mdl,train2predictio[0]), train2predictio[1])   
          pb=proba(mdl,train2predictio[0]);
          echo('predict_proba:',print_evaluation_scores(pb, train2predictio[1]))  
      del(globals()[x],mdl)
  alpow.p=p=print

import scipy
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.neural_network import MLPClassifier
#}{end lib

#}{Flask
from flask import Flask,render_template,url_for,request
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if (request.method == 'POST'):
        message = request.form['message']
        

#}{get model
mdlfile=basedir+'mdl.mdl.zip';
if(not os.path.exists(mlfile)):
    os.system('wget '+mdlfile)
os.system('unzip '+mdlfile)
#load model
mdl=joblib.load(mdlfile);




