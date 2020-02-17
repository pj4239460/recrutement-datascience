#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Problem

#Prédire le prix de vente de lots d'un fruit

## Data

#Un dataset contenat des enregistrements de vente d'un fruit


# In[2]:


#Tout d'abord, lisez le fichier fruit_price.csv
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
df = pd.read_csv('D2FRecrutement/fruit_price/fruit_price.csv')
#J'ai besoin d'observer la structure de la table.
df
#Ce tableau contient les colonnes suivantes:
#Unnamed: 0 : Il s'agit du numéro d'index des données de chaque région，Je n'en ai pas besoin, je peux donc le supprimer
#Date : Ce sont des données très intéressantes, je peux les utiliser, mais je dois les traiter.
#AveragePrice : C'est sont les prix que j'ai besoin d'analyser et de prédire.
#Total Volume, 4046, 4225, 4770, Total Bags,Small Bags,Large Bags,XLarge Bags: des facteurs qu'on peut utiliser.
#type,region: ces sont des variables qualitatives, je peux utiliser One Hot Encoding
#year: year est déjà incluse dans Date


# In[3]:


df.describe()


# In[4]:


#Date
#"Date" est un facteur très utile, mais je ne peux pas l'utiliser directement.
#Puisque nous avons des données de 2015-2018, j'ai converti "Date" en combien de jours se sont écoulés depuis le nouvel an de cette année.
#Cela transformera "date" en une variable quantitative
from datetime import datetime
def date_diff(date):#Il Peut calculer le nombre de jours écoulés depuis le nouvel an
    first_new_year=str(date[0:4])+"-01-01"
    date = datetime.strptime(date, '%Y-%m-%d')
    first_new_year = datetime.strptime(first_new_year, '%Y-%m-%d')
    return (date-first_new_year).days

df['date_newyear_num']=df["Date"].apply(lambda x : date_diff(x))


# Data Visualisation

# In[5]:


#Tout d'abord, je peux utiliser la visualisation des données pour comprendre la relation entre chaque facteur et le prix.
df.plot.scatter(x="Total Volume",y="AveragePrice")
#Ici, je ne vois pas clairement de relation entre le prix et "Total Volume".


# In[6]:


df.plot.scatter(x="Total Bags",y="AveragePrice")
#Je ne vois pas clairement de relation entre le prix et "Total Bags".


# In[7]:


df.groupby('Date')['AveragePrice'].mean().plot(kind='line')
#Je vois que le prix est lié à la "date".


# In[8]:


df.groupby('date_newyear_num')['AveragePrice'].mean().plot(kind='line')


# In[9]:


df.groupby('type')['AveragePrice'].mean().plot(kind='bar')
#Je vois que le prix peut-être lié à la "type".


# In[10]:


df.groupby('year')['AveragePrice'].mean().plot(kind='bar')
#Je ne vois pas clairement de relation entre le prix et "year".


# In[11]:


df.groupby('region')['AveragePrice'].mean().plot(kind='bar')
#Je ne vois pas clairement de relation entre le prix et "region".


# In[12]:


#Construire Une matrice de corrélation
corrMatt = df.corr()
corrMatt


# In[13]:


mask = np.array(corrMatt)
mask[np.tril_indices_from(mask)] = False
fig,ax= plt.subplots()
fig.set_size_inches(15,8)
sn.heatmap(corrMatt, mask=mask,vmax=.8, square=True,annot=True)
#J'utilise Heat Map pour représenter les relations entre divers facteurs et le prix. 
#J'ai trouvé que la plupart des facteurs sont liés à "prix", mais ils ne sont pas très pertinents. 
#Évidemment, je dois faire des traitements sur les variables.


# Nettoyage de données

# In[14]:


df.count()
#Utilisez la fonction de count() pour déterminer s'il existe des valeurs manquantes
#les résultats montrent qu'il n'y a pas de valeurs manquantes dans les données.


# In[15]:


df.describe()


# In[16]:


df


# In[17]:


#Il ressemble que volume total = 4046 + 4225 + 4770 + Total Bags. Cependant, je dois le vérifier.
re_Total_Volume = pd.DataFrame((df["4046"]+df["4225"]+df["4770"]+df["Total Bags"])-df["Total Volume"])
#J'ai construit une nouvelle dataframe "re_Total_Volume", qui stocke la différence entre la somme de 4046, 4225, 4770, Total Bags et "Total Volume"
re_Total_Volume.describe()#résultat des statistiques


# In[18]:


#Je peux voir qu'il y a pas mal de différences entre les deux. df["4046"]+df["4225"]+df["4770"]+df["Total Bags"] et df["Total Volume"]
#Mais cela peut également être dû à certaines erreurs après avoir arrondi les données. 
#Je me concentre donc uniquement sur combien de valeurs dans "re" dont la valeur absolue est supérieure à 1.
re_Total_Volume[(re_Total_Volume[0]> 1) | (re_Total_Volume[0]<-1)]
#Je vois qu'il y a encore des valeurs dans "re" dont la valeur absolue est supérieure à 1. 
#Comme je ne sais pas comment le "Volume total" est calculé, j'ai décidé de le garder.


# In[19]:


#Il ressemble que Total Bags = Small Bags+Large Bags+XLarge Bags. Cependant, je dois le vérifier.
re_Total_Bags = pd.DataFrame((df["Small Bags"]+df["Large Bags"]+df["XLarge Bags"])-df["Total Bags"])
#J'ai construit une nouvelle dataframe "re_Total_Bags"
re_Total_Bags.describe()#résultat des statistiques


# In[20]:


re_Total_Bags[(re_Total_Bags[0]> 1) | (re_Total_Bags[0]<-1)]
#Je vois que la valeur absolue de la différence entre "Total Bags" et "Small Bags + Large Bags + XLarge Bags" est toujours inférieure à 1. 
#Je pense que cela montre que "Total Bags" est approximativement égal à "Small Bags + Large Bags + XLarge Bags", 
#je pense donc que la colonne "Total Bags" peut être supprimée car les informations qu'elle contient sont bien exprimées dans les trois autres colonnes.


# In[21]:


df


# In[22]:


#Pour les trois variables qualitatives "type", "année" et "région", j'utilise One Hot Encoding pour les traiter.
df=pd.get_dummies(df,columns=['type'])
df=pd.get_dummies(df,columns=['year'])
df=pd.get_dummies(df,columns=['region'])


# In[23]:


#Ensuite, j'ai supprimé 'Unnamed: 0', 'Date',"Total Bags" trois colonnes.
df = df.drop(['Unnamed: 0', 'Date','Total Bags'], axis=1)


# In[24]:


df.columns


# Les Valeurs Aberrantes

# In[25]:


#Puisqu'il n'y a pas de valeurs nulles dans les données, j’ai appliqué la méthode des trois écarts-types pour supprimer les Valeurs Aberrantes (Outliers). 
outliers=np.abs(df["AveragePrice"]-df["AveragePrice"].mean()) >(3*df["AveragePrice"].std())
outliers = pd.DataFrame(outliers)
outliers_list = outliers.loc[outliers["AveragePrice"]==True]._stat_axis.values.tolist()


# In[26]:


#131 valeurs aberrantes seront supprimées
len(outliers_list)


# In[27]:


df = df.drop(index=outliers_list)


# Plot Kde

# In[28]:


#Il s'agit d'un problème de régression, donc si la valeur cible suit la loi normale, cela fonctionnera bien pour de nombreux modèles.
df['AveragePrice'].plot(kind = 'kde')
#Ce graphique n'est pas une bonne distribution de la loi normale, il peut donc être difficile d'utiliser certains types de modèle linéaire.


# In[29]:


df.to_csv("fruit_price_new.csv")
df


# Machine Learning

# In[30]:


df = pd.read_csv("fruit_price_new.csv", index_col=0)
df


# In[31]:


#Je commence la partie apprentissage automatique, j'utilise "AveragePrice" comme y et tous les autres facteurs comme X
X_feature = [col for col in df.columns.values if col != "AveragePrice"]
y_feature = ["AveragePrice"]
X = df.loc[:,X_feature]
y = df.loc[:,y_feature]


# In[32]:


#Je divise au hasard le dataset en trainset et testset, le rapport 7: 3
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[33]:


#Comme je l'ai mentionné précédemment, il peut être difficile d'utiliser certains types de modèle linéaire. 
#Par conséquent, j'ai choisi deux méthodes d'apprentissage automatique: Random Forest Regression et Gradient Boosting Regression

from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from tqdm import *
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.externals import joblib


# In[34]:


#J'utilise RMSE(root mean square error) et RMSLE(root mean squared logarithmic error) pour évaluer le modèle

def rmsle(y_test, y_pred):
    return np.sqrt(mean_squared_log_error(y_test, y_pred))

def rmse(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test,y_pred))


# Random Forest Regressor

# In[35]:


#J'utilise GridSearch pour optimiser les hyperparameters 
#(je n'ai pas utilisé beaucoup d'hyperparameters en raison des performances limitées de mon ordinateur!!!)
estimator = RandomForestRegressor()
parameters = { 
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5, 10],
    'n_estimators': [100, 1000, 2000],
    'random_state' : [0],
    'n_jobs' : [-1]
}
rmsle_scorer = metrics.make_scorer(rmsle, greater_is_better=False)
grid_RFR = GridSearchCV( estimator,
                          param_grid=parameters,
                          scoring = rmsle_scorer,
                          cv=5)

grid_RFR.fit(X=X_train,y=y_train.to_numpy().ravel())


# In[36]:


#Afficher tous les parameters et RMSLE correspondant pour sélectionner les meilleurs parameters
means = grid_RFR.cv_results_['mean_test_score']
params = grid_RFR.cv_results_['params']
for mean,param in zip(means,params):
    print("%f  with:   %r" % (mean,param))
print(grid_RFR.best_params_)
print(grid_RFR.best_score_)


# In[37]:


pre=grid_RFR.predict(X_test)
print('RMSLE：',rmsle(y_test, pre))


# In[38]:


#J'utilise les meilleurs paramètres pour entraîner le modèle
RFR = RandomForestRegressor(min_samples_leaf = grid_RFR.best_params_["min_samples_leaf"],
                            min_samples_split = grid_RFR.best_params_["min_samples_split"],
                            n_estimators = grid_RFR.best_params_["n_estimators"],
                            random_state=0, 
                            n_jobs=-1)
RFR.fit(X=X_train,y=y_train.to_numpy().ravel())


# In[39]:


#J'ai utilisé le modèle pour prédire le testset, puis j'ai exprimé les erreurs en utilisant RMSE et RMSLE.
pre=RFR.predict(X_test)
print("RMSE of Random Forest Regression: ", rmse(y_test, pre))
print("RMSLE of Random Forest Regression: ", rmsle(y_test, pre))


# In[40]:


#RMSE 0.1188 RMSLE 0.0487 D'après la figure suivante, il semble que les résultats du modèle sont assez bons. 
x_axix = list(range(100))
plt.figure(figsize=(10,8))
plt.title('Random Forest Regression')
plt.plot(x_axix, pre[0:100], color='green', label='Prediction')
plt.plot(x_axix, y_test[0:100], color='red', label='Reality')
plt.xlabel('Sample index')
plt.ylabel('AveragePrice')
plt.show()


# In[41]:


#Enregistrer le modèle "RFR.pkl" 
with open('RFR.pkl', 'wb') as f:
    joblib.dump(RFR, 'RFR.pkl')


# Gradient Boosting Regressor

# In[42]:


#J'utilise GridSearch pour optimiser les hyperparameters 
#(je n'ai pas utilisé beaucoup d'hyperparameters en raison des performances limitées de mon ordinateur!!!)
estimator = GradientBoostingRegressor()
parameters = { 
    'n_estimators':[100,500,1000],
    'learning_rate': [0.1,0.05,0.02],
    'max_depth':[4,3,2],
    'min_samples_leaf':[1,2,3]
}
rmsle_scorer = metrics.make_scorer(rmsle, greater_is_better=False)
grid_GBR = GridSearchCV( estimator,
                          param_grid=parameters,
                          scoring = rmsle_scorer,
                          cv=5)

grid_GBR.fit(X=X_train,y=y_train.to_numpy().ravel())


# In[43]:


#Afficher tous les parameters et RMSLE correspondant pour sélectionner les meilleurs parameters
means = grid_GBR.cv_results_['mean_test_score']
params = grid_GBR.cv_results_['params']
for mean,param in zip(means,params):
    print("%f  with:   %r" % (mean,param))
print(grid_GBR.best_params_)
print(grid_GBR.best_score_)


# In[44]:


pre=grid_GBR.predict(X_test)
print('RMSLE：',rmsle(y_test, pre))


# In[45]:


#J'utilise les meilleurs paramètres pour entraîner le modèle
GBR = GradientBoostingRegressor(n_estimators = grid_GBR.best_params_["n_estimators"],
                               learning_rate = grid_GBR.best_params_["learning_rate"],
                               max_depth = grid_GBR.best_params_["max_depth"],
                               min_samples_leaf = grid_GBR.best_params_["min_samples_leaf"])
GBR.fit(X=X_train,y=y_train)


# In[46]:


#J'ai utilisé le modèle pour prédire le testset, puis j'ai exprimé les erreurs en utilisant RMSE et RMSLE.
pre=GBR.predict(X_test)
print("RMSE of Gradient Boosting Regression: ", rmse(y_test, pre))
print("RMSLE of Gradient Boosting Regression: ", rmsle(y_test, pre))


# In[49]:


#RMSE 0.1142 RMSLE 0.0463 D'après la figure suivante, il semble que les résultats du modèle sont assez bons.
#Et les résultats de Gradient Boosting Regression(RMSE-0.1142,RMSLE-0.0463) sont meilleurs que les résultats de Random Forest Regression(RMSE-0.1188,RMSLE-0.0487)
#donc le meilleur modèle que j'ai obtenu est Gradient Boosting Regression
x_axix = list(range(100))
plt.figure(figsize=(10,8))
plt.title('Gradient Boosting Regression')
plt.plot(x_axix, pre[0:100], color='green', label='Prediction')
plt.plot(x_axix, y_test[0:100], color='red', label='Reality')
plt.xlabel('Sample index')
plt.ylabel('AveragePrice')
plt.show()


# In[48]:


#Enregistrer le modèle "GBR.pkl" 
with open('GBR.pkl', 'wb') as f:
    joblib.dump(GBR, 'GBR.pkl')


# Les facteurs les plus importantes

# In[50]:


#Charger le modèle "GBR.pkl" 
GBR = joblib.load("GBR.pkl")


# In[51]:


#le classement de l'influence de divers facteurs sur le prix.
indies = np.argsort(GBR.feature_importances_, kind='heapsort')[::-1]
for index in indies:
    print({X_feature[index]:GBR.feature_importances_[index]})


# In[ ]:




