#prod:git clone https://github.com/ben74/stackoverflow_tagprediction;cd stackoverflow_tagprediction;git pull;python3 cassandra.py 2>&1 | tee cassandra.log
#dev: python3 ~/home/_cassandra/cassandra.py
#todo : inclure l'import des modèles ci dessous
#todo : ajouter fonctions de cleaning sur le texte en input
#modèle final : aller sur stack overflow au hazard ou via sql explorer et tester
#pip install flask sklearn;
#rm bestQuickModel.tgz;rm bestQuickModel.pickle;cd ~/home/python;py3 cassandra.py;#has 344 tags, not 358
#gad *.tgz;gu;git push -f
#}{Required modules installation
#}{Alpow
import alpow;from alpow import *
message('cassandra online')
sftp['cd']='stack5'
alpow.webRepo='https://1.x24.fr/a/jupyter/'
#dont store credentials within the repository
if os.path.exists('credentials.py'):
    import credentials;
    alpow.sftp=credentials.sftp
    #p(alpow.sftp);
    p('sftp online:',ftpls()[0],'\n\n\n')
else:
    alpow.useFTP=False;
    sendimages2ftp=0
#}{
import numpy as np
import nltk
from nltk.tokenize import ToktokTokenizer
np.random.seed(1983)

nltk.download('stopwords')
alpow.stop_words=nltk.corpus.stopwords.words('english')
alpow.lemma=nltk.WordNetLemmatizer()
alpow.token=ToktokTokenizer()

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
        elif type(globals()[fn])==pd.DataFrame:
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
  
def FPCP(fn, data):
    os.system('rm -f ' + fn + '.pickle ' + fn + '.pickle.zip')
    f = open(fn + '.pickle', 'wb')
    pickle.dump(data, f, protocol=-1)
    
def fgcp(fn):
    #p('true is ',fn)
    fn = fn + '.pickle'
    getFile(fn)
    if verbose:
        p('size:', fn, ' : ', os.path.getsize(fn) / 1024 / 1024)
    if(os.path.isfile(fn)):
        gc.disable()
        data = open(fn, "rb");
        ret = rewriteClass(data).load()
        data.close()
        gc.enable()
        return ret
    return 0
alpow.fgcp=fgcp
    
#ModuleNotFoundError: No module named 'sklearn.linear_model._logistic
class rewriteClass(pickle.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        a=module.split('.')        
        if((a[-1][0]=='_') & (a[-1] not in ['_pickle'])):
#sur une erreur liée à la déserialization, s'assurer d'avoir les mêmes versions entre deux machines !!        
#https://github.com/RaRe-Technologies/gensim/issues/2602        
#pip uninstall numpy;pip install numpy==1.18.2
            #p(a[-1],a[-1][0])
            echo('-rewriting class:','.'.join(a))
            del(a[-1])            
            renamed_module='.'.join(a)
        
        if module == "sklearn.linear_model._logistic":
            renamed_module = "sklearn.linear_model"
        return super(rewriteClass, self).find_class(renamed_module, name)
          
#}{LDA
def ldaResults(x,best_lda_model,vectorizer_train,df_topics_tags_norm,nb=5,titleWeight=4):
  input=(x['title']+' ')*titleWeight + x['body']#is a serie  
  vectorized=vectorizer_train.transform(input)
  topic_distrib_pred = best_lda_model.transform(vectorized)
  probabilityPerTag = (df_topics_tags_norm * topic_distrib_pred).sum(axis=1)
  #echo(x['title'].values[0]," -- ",x['body'].values[0],"\n",'tags : ',x['tags'].values[0])
  top10=arsort(probabilityPerTag)[:nb]
  #display(top10)
  tags=[]
  for i in top10[:nb]:
    tags+=[i[0]]
  return tags

#}{TFIDF
def text2tfidf(mdl,mlbTrain,TFvectorizer,df,mindf=5.5e-05,maxdf=0.0080,maxfeatures=1000,titleWeight=4,nb=1):  
  inputx=df['title'][:nb]*titleWeight+ ' '+df['body'][:nb]
  TFIDF=TFvectorizer.transform(inputx)
  predictions=mdl.predict(TFIDF)
  tags=mlbTrain.inverse_transform(predictions)
  return tags

#display(x['title'].values[0],x['body'].values[0],x['tags'].values[0],'=>',tags)#

#}{Bag of words
def my_bag_of_words(text, words_to_index, dict_size):
  result_vector = np.zeros(dict_size)
  keys= [words_to_index[i] for i in text.split(" ") if i in words_to_index.keys()]
  result_vector[keys]=1
  return result_vector
 
def EvaluateBagModel(mdl,x,WORDS_TO_INDEX,titleWeight=4,pca2=False,nb=1):
  from itertools import chain 
  inputx=(x['title'][:nb]+' ')*titleWeight+x['body'][:nb]
  MBW=[]  
  for text in inputx:#single
    #text=' '.join(inputCleaner(text));
    MBW+=[my_bag_of_words(text, WORDS_TO_INDEX, dictsize)]
#cibler le nombre de dimensions finales de la PCA !!
#avec les mêmes paramètres devant retourner le même nombre de dimensions ? => retourne matrice vide => bof bof
  if pca2:
    MBW2=pca2.fit_transform(MBW)
    sumVariance=round(list(pca2.explained_variance_ratio_.cumsum()).pop(),2)
    p('NbDimensions:',MBW.shape[1],', sum variance:',sumVariance)
  else:
    MBW=np.array(MBW)
#assert(len(MBW[0]),'/',len(bag_train[0]))#is 3000 / 3000  => fits
#ValueError: X has 3000 features per sample; expecting 1011 ( nb dimensions de réduction pca )
  predictions=mdl.predict(np.array(MBW))  
  #p('dictsize:',dictsize,',WORDS_TO_INDEX:',len(WORDS_TO_INDEX),';mbwl:',len(MBW[0]))
  assert(len(mlbTrain.classes_)==len(predictions[0])) 
  predict_proba=proba(mdl,MBW)
  assert(len(mlbTrain.classes_)==len(predict_proba[0])) 
#modèle de base plus sensible  
  tags1=mlbTrain.inverse_transform(predictions)
  tags1=list(chain.from_iterable(tags1)) 
#predict_proba: moins de rappel  
  tags2=mlbTrain.inverse_transform(predict_proba)
  tags2=list(chain.from_iterable(tags2)) 
  p(tags1,' <> Predict proba:',tags2)
  
  return tags1,tags2

def g(x):
    return globals()[x]

def proba(_mdl,x,seuil=0,minus=1,top=2):
  import math
  global probasPerClass,topTags,QuestionToTags,probaIdPerQuestion,probaTagId2,ypredictions,nbTags,emptyMatrix
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
    p('not simple')
    while(i<nbTags-minus):#1370
      if(len(probasPerClass[i][0])>1):
        probaTagId2[i]=probasPerClass[i][:,1]
#multiple predictions        
      else:
        probaTagId2[i]=[0]*rows;
      i+=1

  probaIdPerQuestion=pd.DataFrame(probaTagId2)
#¤todo:si seuil > tant
  if seuil:
    probaIdPerQuestion=probaIdPerQuestion[probaIdPerQuestion>seuil]

  topTags = probaIdPerQuestion.apply(lambda x: pd.Series(np.concatenate([x.nlargest(top).index.values])), axis=1)  
  QuestionToTags=topTags.T.to_dict()#{0: {0: 203, 1: 247}}
  emptyMatrix = [0] * nbTags

#0 : 158,139,1 : 282,283,2 : 234,52,
  ypredictions=[]  
  for i in list(QuestionToTags.keys()):
    #print(i, end=' : ')
    y = emptyMatrix.copy()
    for j in list(QuestionToTags[i].values()):
      if(not math.isnan(j)):
        y[int(j)] = 1        
    ypredictions += [y]

  return np.array(ypredictions)    

def inputCleaner(x):
    x=x.lower()#based on string then
    return ''.join(extractionMots(x))  
  
dtitle='python confused question'
dbody='''
- Some text about flask and class
- this is the text within the html body contents like mysql query request having im going to write more mysql related database stuff
- would adding some mysql database related queries and words would rise the tag probability for it ? => True
'''
newdf=pd.DataFrame({'title':[dtitle],'body':[dbody],'tags':''})   

#relatifs au modèle :: ValueError: Expected indicator for 40 classes, but got 358
nb=1
#p=nf;#null for load prints
load('mlbTrain')
assert(len(mlbTrain.classes_)==358)

load('best_lda_model,vectorizer_train,df_topics_tags_norm')
echo('lda results:',ldaResults(newdf,best_lda_model,vectorizer_train,df_topics_tags_norm,nb=2))
#lda is perfect :)
echo(len(mlbTrain.classes_))
##ValueError: Expected indicator for 358 classes, but got 345 ==> WHY ??? wasn't re-trained at all  <<< just loaded

if False:
    p('_'*180)
    k3='2000_2000';dictsize=2000;mdlname='short';
    load(mdlname+',bagOfWords_'+k3)
    echo(EvaluateBagModel(g(mdlname),newdf,g('bagOfWords_'+k3)))

if False:
    p('_'*180)
    k3='10000_2000';dictsize=2000;mdlname='bqm';
    load(mdlname+',bagOfWords_'+k3)
    echo(EvaluateBagModel(g(mdlname),newdf,g('bagOfWords_'+k3)))

if True:
    p('_'*180)
    k3='50000_5000';dictsize=5000;mdlname='bestModel2';
    load(mdlname+',bagOfWords_'+k3);echo(mdlname);
    echo(EvaluateBagModel(g(mdlname),newdf,g('bagOfWords_'+k3)))








p=print;#resume


#mlbTrain=mlbTrain348classes
#ValueError: Expected indicator for 40 classes, but got 345



# 
#}{Falsk

from flask import Flask,render_template,url_for,request
app = Flask(__name__)#, root_path = '/'

@app.route('/')
def home():
    title=dtitle
    body=dbody
    return render_template('main.html', title=title,body=body)

@app.route('/',methods=['POST'])
def predict():
    if (request.method == 'POST'):        
        title = request.form['title']        
        body = request.form['body']                        
        newdf = pd.DataFrame({'title':[inputCleaner(title)],'body':[inputCleaner(body)],'tags':''})
        tags1,tags2=EvaluateBagModel(globals()[mdlname],newdf,g('bagOfWords_'+k3))
        tags1=' , '.join(tags1)
        tags2=' , '.join(tags2)
        ldatags=' , '.join(ldaResults(newdf,best_lda_model,vectorizer_train,df_topics_tags_norm,nb=2))
        #tags=text2tfidf(globals()[k+mdln],globals()[k+'mlbTrain'],globals()[k+'TFvectorizer'],newdf)  
    return render_template('main.html', tags1 = tags1, tags2 = tags2, ldatags = ldatags,title=title, body=body, postdata = 1)

if __name__ == '__main__':
	app.run(host='0.0.0.0',port=8080,debug=True)       


