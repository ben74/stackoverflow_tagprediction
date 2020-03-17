#todo:inclure l'import des modèles ci dessous

#pip install flask
from flask import Flask,render_template,url_for,request
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('main.html')

@app.route('/predict',methods=['POST'])
def predict():
    if (request.method == 'POST'):
        title = request.form['title']        
        body = request.form['body']        
        tags='tags: '+title+' '+body
    return render_template('main.html', tags = tags, postdata = 1)

if __name__ == '__main__':
	app.run(debug=True)       

#}{Alpow
import alpow;from alpow import *

newdf=pd.DataFrame({'title':['python question confused'],'body':['''
- Some text about flask and class
- this is the text within the html body contents like mysql query request having im going to write more mysql related database stuff
- would adding some mysql database related queries and words would rise the tag probability for it ? => True
'''],'tags':''})
#}{LDA
def ldaResults(x,best_lda_model,vectorizer_train,df_topics_tags_norm,nb=5):
  vectorized=vectorizer_train.transform(x['title']+' '+x['body'])
  topic_distrib_pred = best_lda_model.transform(vectorized)
  probabilityPerTag = (df_topics_tags_norm * topic_distrib_pred).sum(axis=1)
  echo(x['title'].values[0]," -- ",x['body'].values[0],"\n",'tags : ',x['tags'].values[0])
  top10=arsort(probabilityPerTag)[:10]
  #display(top10)
  tags=[]
  for i in top10[:nb]:
    tags+=[i[0]]
  return tags
  
load('best_lda_model,vectorizer_train,df_topics_tags_norm') 
echo('\nLda result tags => ',ldaResults(newdf,best_lda_model,vectorizer_train,df_topics_tags_norm))

#}{TFIDF
def text2tfidf(mdl,mlbTrain,TFvectorizer,df,mindf=5.5e-05,maxdf=0.0080,maxfeatures=1000,titleWeight=4,nb=1):  
  inputx=df['title'][:nb]*titleWeight+ ' '+df['body'][:nb]
  TFIDF=TFvectorizer.transform(inputx)
  predictions=mdl.predict(TFIDF)
  tags=mlbTrain.inverse_transform(predictions)
  return tags

mdln='model_ovr_linearsvc_tfidf'
k='iterativeTrainSplit_2000_1000_'
load(k+mdln+','+k+'mlbTrain,'+k+'TFvectorizer')
tags=text2tfidf(globals()[k+mdln],globals()[k+'mlbTrain'],globals()[k+'TFvectorizer'],newdf)
display(x['title'].values[0],x['body'].values[0],x['tags'].values[0],'=>',tags)#

#}{Bag of words
def EvaluateBagModel(mdl,x,WORDS_TO_INDEX,titleWeight=4,pca2=False):
  inputx=x['title']*titleWeight+ ' '+x['body'][:nb]
  MBW=[]
  for text in inputx:
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
  tags=mlbTrain.inverse_transform(predictions)
  return tags  
  
k3='5000_3000'  
mdlname='bag_5000_3000_model_ovr_linearsvc_test1_bag'
load('bagOfWords,k3,'+mdlname)
echo(EvaluateBagModel(globals()[mdlname],newdf,bagOfWords[k3]))  
