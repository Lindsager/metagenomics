#!/usr/bin/env python
# coding: utf-8

# #### CSE41204 Final Project - Predicting Ocean Metagenomic Sample Biome and Depth with Machine Learning<br>
# DATA PREPARATION<br>
#     • import .csv files for sequencing reads and metadata<br>
#     • Use pandas to clean, prep and merge data frames for predictive classification modeling<br> 
#     
# MACHINE LEARNING<br>
#     • Test 9 ML models each for sampling depth prediction and geographic biome prediction<br>
#     • Report accuracy score, classification report and confusion matrix (heatmap) for ML model with highest accuracy<br> 
#     
# SAMPLE DIVERSITY VISUALIZATION<br>
#     • Use Scikit-Bio package to calculate beta diversity<br>
#     • Create PCoA scatter plot and beta-diversity box-and-whisker plots

# In[1]:


import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import skbio
from skbio.diversity import alpha_diversity
from skbio.diversity import beta_diversity
from skbio.stats.distance import mantel
import skbio.stats.ordination
from skbio.stats.ordination import pcoa
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from matplotlib import cm
import pylab as P
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


# #### DATA PREPARATION - OTUtable.CSV
# Read in OTU table .csv file<br>
# Rename and concatenate separated taxa columns to become single column, set_index as new concat 'OTU_ID' column<br>
# Filter out taxa organisms with <100 total reads across samples (considered seq noise)

# In[3]:


OTUtable=pd.read_csv('miTAG.taxonomic.profiles.release.csv')

OTUtable.columns=OTUtable.columns.str.replace(' ','_').str.lower()
special_char = ")("
for x in special_char:
    OTUtable.columns = OTUtable.columns.str.replace(x,'',regex=True)

for col in OTUtable.columns[:6]:
    x = str(OTUtable.columns.get_loc(col))
    OTUtable[col] = 'D_'+ x +'__'+ OTUtable[col].astype(str) + ';'
    
OTUtable['OTU_ID'] = OTUtable['domain']+OTUtable['phylum']+OTUtable['class']+OTUtable['order']+OTUtable['family']+OTUtable['genus']
drop_cols = OTUtable.columns[:6]
OTU1 = OTUtable[OTUtable.columns.drop(drop_cols)]
OTU1 = OTU1.set_index('OTU_ID')

OTU1['sum'] = OTU1.sum(axis=1)
OTU2 = OTU1.loc[OTU1['sum'] > 100]
del OTU2['sum']
OTU3 = OTU2.transpose()
OTU3.columns.name = 'sampleid'
OTU3


# #### DATA PREPARATION - METDATA.CSV
# Read in metadata .csv file<br>
# Keep only columns for 'sample_ID', 'longhurst_biome', and 'sampling_depth'

# In[4]:


metadata=pd.read_csv('metadata.csv')

metadata.columns=metadata.columns.str.replace(' ','_').str.lower()
special_char = ")(][-"
for x in special_char:
    metadata.columns = metadata.columns.str.replace(x,'',regex=True)
    metadata['environmental_feature'] = metadata['environmental_feature'].str.replace(x,'',regex=True)
    
metadata = metadata.rename(columns={'sample_label_tara_station#_environmentalfeature_sizefraction': 'sampleID','environmental_feature':'sampling_depth','marine_pelagic_biomes_longhurst_2007':'longhurst_biome'})
metadata['sampleid'] = metadata['sampleid'].str.lower()
metadata['sampling_depth'] = metadata['sampling_depth'].str[:3]
metadata['longhurst_biome'] = metadata['longhurst_biome'].str.strip().str.lower().str.replace(' ','_',regex=True)

md1 = metadata[['sampleid','sampling_depth','longhurst_biome']]
md1 = md1.set_index('sampleid')
md1['longhurst_biome'].value_counts()


# #### DATA PREPARATION - MERGE OTU DATA WITH METADATA (ML MODELS)
# For ML modeling - Merge 'OTU1' and 'metadata' on index of 'sample_ID'. Drop any extra samples from metadata that don't have taxa data (ie < 0.22um samples)<br>
# For diversity plotting - save sampleIDs in a list and format values as x,y array (required by scikit-bio)

# In[5]:


ml_data = md1.join(OTU3, how='left')
ml_data = ml_data.dropna()

### for diversity metrics and plots 
div_data = OTU2.T.values
sampleIDs = OTU2.columns
div_depth = ml_data['sampling_depth'].values


# #### DATA PREPARATION - DROPPING LOW ABUNDANCE SAMPLES
# Setting up 'ml_data' table for modeling of sampling depth prediction and then modeling of geographical biome prediction<br>
#     Sampling depth = Only 4/139 samples belong to 'MIX'. Dropping these from the dataset<br>
#     Longhurst Biome = Only 4/139 samples belong to 'Polar_Biome'. Dropping these from the dataset

# In[6]:


ml_depth = ml_data.drop(['longhurst_biome'], axis=1)
ml_biome = ml_data.drop(['sampling_depth'], axis=1)

ml_depth['sampling_depth'].value_counts() 
ml_biome['longhurst_biome'].value_counts()


# In[7]:


#MIX only has 4 samples, so we drop this condition
ml_depth = ml_depth[ml_depth.sampling_depth != 'MIX']
ml_depth['sampling_depth'].value_counts()


# In[8]:


ml_biome = ml_biome[ml_biome.longhurst_biome != 'polar_biome']
ml_biome['longhurst_biome'].value_counts()


# #### SAMPLING DEPTH - ML MODEL TRAINING
# Creating the test and training data sets for 'sampling_depth'

# In[9]:


array = ml_depth.values
Xa = ml_depth.iloc[:,1:]
Ya = ml_depth[['sampling_depth']]
validation_size =  0.2
seed = 7

Xa_train, Xa_validation, Ya_train, Ya_validation =  train_test_split(Xa,Ya, test_size=0.2, random_state=7, shuffle=True)


# #### SAMPLING DEPTH - ML PREDICTIVE MODELING
# 
# (Listed in project proposal)<br>
# • Logistic Regression (LR). ---------------------------> Linear<br>
# • Elastic Net (EN) ----------------------------------------> Linear<br>
# • Support Vector Machines (SVM) ------------------> Non-Linear<br>
# • Random Forest Classifier (RF) --------------------> Ensemble<br>
# • MLP Classifier (MLP) ---------------------------------> Neural-Network<br>
# 
# 
# (Additional models from Iris .ipynb demo)<br>
# • Linear Discriminant Analysis (LDA) ---------------> Linear<br>
# • k-Nearest Neighbors (KNN) ------------------------> Non-Linear<br>
# • Classification and Regression Trees (CART)---> Non-Linear<br>
# • Gaussian Naive Bayes (NB) ------------------------> Non-Linear

# In[10]:


models = []
models.append(('LR', LogisticRegression(max_iter=1100)))
models.append(('SVM', SVC(kernel='poly', C=1)))
models.append(('RF', RandomForestClassifier(max_depth=6, n_estimators=100)))
models.append(('MLP', MLPClassifier(hidden_layer_sizes=2, learning_rate_init=.3, momentum=.2, max_iter=10000)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))

# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=10, random_state=seed, shuffle=True)
    cv_results = cross_val_score(model, Xa_train, Ya_train.values.ravel(), cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# #### SAMPLING DEPTH - MODEL TESTING RESULTS
# 
# Plot the results from model testing

# In[11]:


fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()


# #### SAMPLING DEPTH - SUMMARIZE RESULTS
# 
# The logistic regression model with elastic net parameter tuning was the most accurate.
# 
# Print the accuracy score, classification report and confusion matrix for the logistic regression model.

# In[12]:


rf = RandomForestClassifier(max_depth=5, n_estimators=100)
rf.fit(Xa_train, Ya_train.values.ravel())
Ya_prediction = rf.predict(Xa_validation)

score = accuracy_score(Ya_validation, Ya_prediction)
print(f'Accuracy: {score}')

print(classification_report(Ya_validation, Ya_prediction))

confmatrix = confusion_matrix(Ya_validation, Ya_prediction)
print(confmatrix)

plt.figure(figsize = (10,7))
sns.heatmap(confmatrix, annot=True, fmt='', cmap='Reds')
plt.xlabel('Predicted')
plt.ylabel('Actual')
# 0-DCM, 1-MES, 2-SRF


# #### BIOME CLASSIFICATION
# 
# Creating the test and training data sets for 'longhurst_biome'
# ![image.png](attachment:image.png)
#     
# Coastal, Trades (Tropical), Westerlies (Temperate), or Polar

# In[13]:


array = ml_biome.values
Xb = ml_biome.iloc[:,1:]
Yb = ml_biome[['longhurst_biome']]
validation_size =  0.2
seed = 7

Xb_train, Xb_validation, Yb_train, Yb_validation =  train_test_split(Xb,Yb, test_size=0.2, random_state=7, shuffle=True)


# #### BIOME CLASSIFICATION - ML PREDICTIVE MODELING
# 
# Building Models for Geographical Biome Classification ('longhurst_biome')<BR>
# (Listed in project proposal)<BR>
# • Logistic Regression (LR). ---------------------------> Linear<BR>
# • Elastic Net (EN) ----------------------------------------> Linear<BR>
# • Support Vector Machines (SVM) ------------------> Non-Linear<BR>
# • Random Forest Classifier (RF) --------------------> Ensemble<BR>
# • MLP Classifier (MLP) ---------------------------------> Neural-Network<BR>
# 
# (Additional models from Iris .ipynb demo)<BR>
# • Linear Discriminant Analysis (LDA) ---------------> Linear<BR>
# • k-Nearest Neighbors (KNN) ------------------------> Non-Linear<BR>
# • Classification and Regression Trees (CART)---> Non-Linear<BR>
# • Gaussian Naive Bayes (NB) ------------------------> Non-Linear

# In[14]:


models = []
models.append(('LR', LogisticRegression(max_iter=1100)))
models.append(('SVM', SVC(kernel='poly', C=1)))
models.append(('RF', RandomForestClassifier(max_depth=6, n_estimators=1000)))
models.append(('MLP', MLPClassifier(hidden_layer_sizes=2, learning_rate_init=.3, momentum=.2, max_iter=10000)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))

# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=10, random_state=seed, shuffle=True)
    cv_results = cross_val_score(model, Xb_train, Yb_train.values.ravel(), cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# #### BIOME CLASSIFICATION - MODEL TESTING RESULTS
# 
# Plot the results from model testing

# In[15]:


fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()


# #### BIOME CLASSIFICATION - SUMMARIZE RESULTS
# 
# The logistic regression model with elastic net parameter tuning was the most accurate.<BR>
# Print the accuracy score, classification report and confusion matrix for the logistic regression model.
# 

# In[16]:


logreg = LogisticRegression(max_iter=1100)
logreg.fit(Xb_train, Yb_train.values.ravel())
Yb_prediction = logreg.predict(Xb_validation)

score = accuracy_score(Yb_validation, Yb_prediction)
print(f'Accuracy: {score}')

print(classification_report(Yb_validation, Yb_prediction))

confmatrix = confusion_matrix(Yb_validation, Yb_prediction)
print(confmatrix)

plt.figure(figsize = (10,7))
sns.heatmap(confmatrix, annot=True, fmt='', cmap='Reds')
plt.xlabel('Predicted')
plt.ylabel('Actual')
# 0-coastal_biome, 1-trades_biome(tropical), 2-westerlies_biome(temperate)


# #### DIVERSITY CALCULATIONS AND PLOTS

# In[17]:


OTU3
div_data
sampleIDs


# #### BETA DIVERSITY : BRAY-CURTIS

# In[18]:


bc_dm = beta_diversity('braycurtis',div_data,sampleIDs)
bc_df = bc_dm.to_data_frame()
bc_df.index.rename('sampleid', inplace=True)
bc_dm


# In[19]:


bc_PCoA = pcoa(bc_dm)
bc_PCoA_prps=bc_PCoA.proportion_explained
bc_PCoA_mat=bc_PCoA.samples[['PC1','PC2']]
scatter_PCoA = sns.scatterplot(data=bc_PCoA_mat, x='PC1', y='PC2', legend=False)


# #### SAMPLING DEPTH - BETA DIVERSITY & BOX PLOTS
# 
# 

# In[22]:


ml_depth1 = ml_depth.reset_index()
md_depth = ml_depth1[['sampleid','sampling_depth']]

def bc_depth(depth_class):
    df = ml_depth.loc[ml_depth['sampling_depth'] == depth_class]
    df1 = df.drop(columns='sampling_depth', axis=1)
    depth_values = df1.values
    depth_sampleid = df1.T.columns
    
    depth_bc = beta_diversity('braycurtis',depth_values,depth_sampleid)
    depth_bc = depth_bc.to_data_frame()
    depth_bc.index.rename('sampleid', inplace=True)
    depth_bc = depth_bc.merge(md_depth, on='sampleid', how='right')
    depth_bc = depth_bc[depth_bc.sampling_depth == depth_class]
    depth_bc = depth_bc.drop(columns=['sampleid','sampling_depth'], axis=0)
    depth_list = depth_bc.values.tolist()
    return depth_list
    

SRF_values = bc_depth('SRF')
DCM_values = bc_depth('DCM')
MES_values = bc_depth('MES')

beta_depths = [SRF_values, DCM_values, MES_values]


# In[23]:


fig = plt.figure(1, figsize=(9, 6))
ax = fig.add_subplot(111)
bp = ax.boxplot(beta_depths)
fig.savefig('fig1.png', bbox_inches='tight')

bp = ax.boxplot(beta_depths, patch_artist=True)
for box in bp['boxes']:
        box.set( color='#aeb370', linewidth=2)
        box.set( color='#aeb370', linewidth=2)
        box.set( color='#aeb370', linewidth=2)
for whisker in bp['whiskers']:
    whisker.set(color='#aeb370', linewidth=2)
for cap in bp['caps']:
    cap.set(color='#aeb370', linewidth=2)   
for median in bp['medians']:
    median.set(colorb37570', linewidth=2)
for flier in bp['fliers']:
    flier.set(marker='o', color='#e7298a', alpha=0.5)
    
ax.set_xticklabels(['SRF', 'DCM', 'MES'])
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
ax.set_title('Sampling Depth - Beta Diversity')
ax.set_xlabel('Sampling Depth')
ax.set_ylabel('Beta Diversity')


# #### BIOME CLASSIFICATION - BETA DIVERSITY & BOX PLOTS

# In[32]:


ml_biome1 = ml_biome.reset_index()
md_biome = ml_biome1[['sampleid','longhurst_biome']]

def biome_bc(biome_class):
    biome_df = ml_biome.loc[ml_biome['longhurst_biome'] == biome_class]
    biome_df = biome_df.drop(columns='longhurst_biome', axis=1)
    biome_values = biome_df.values
    biome_sampleid = biome_df.T.columns
    
    biome_bc = beta_diversity('braycurtis',biome_values,biome_sampleid)
    biome_bc = biome_bc.to_data_frame()
    biome_bc.index.rename('sampleid', inplace=True)
    biome_bc = biome_bc.merge(md_biome, on='sampleid', how='right')
    biome_bc = biome_bc[biome_bc.longhurst_biome == biome_class]
    biome_bc = biome_bc.drop(columns=['sampleid','longhurst_biome'], axis=0)
    biome_list = biome_bc.values.tolist()
    return biome_list

trades_values = biome_bc('trades_biome')
coast_values = biome_bc('coastal_biome')
west_values = biome_bc('westerlies_biome')

beta_biomes = [trades_values, coast_values, west_values]


# In[33]:


fig = plt.figure(1, figsize=(9, 6))
ax = fig.add_subplot(111)
bp = ax.boxplot(beta_biomes)
fig.savefig('fig1.png', bbox_inches='tight')

bp = ax.boxplot(beta_biomes, patch_artist=True)
for box in bp['boxes']:
        box.set( color='#70aeb3', linewidth=2)
        box.set( color='#70aeb3', linewidth=2)
        box.set( color='#70aeb3', linewidth=2)
for whisker in bp['whiskers']:
    whisker.set(color='#70aeb3', linewidth=2)
for cap in bp['caps']:
    cap.set(color='#70aeb3', linewidth=2)   
for median in bp['medians']:
    median.set(color='#b37570', linewidth=2)
for flier in bp['fliers']:
    flier.set(marker='o', color='#e7298a', alpha=0.5)
    
ax.set_xticklabels(['Trades Biome', 'Coastal Biome', 'Westerlies Biome'])
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
ax.set_title('Longhurst Biome Classification - Beta Diversity')
ax.set_xlabel('Biome Classification')
ax.set_ylabel('Beta Diversity')

