########################## CREDIT DEFAULT PREDICTION ##########################
################################# TEAM 08 #####################################


########## ADITYA CHUGH
########## YASHASWINI REDDY TERALA

########## IMPORTING NECESSARY LIBRARIES

######## BASIC LIBRARIES

import numpy as np
import pandas as pd
import seaborn as sns
import gc

######## VISUALIZATION TOOLS

import matplotlib.pyplot as plt
import matplotlib.colors
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

######## DATA PROCESSING LIBRARIES

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

######## MODELING LIBRARIES

import optuna
import shap
import itertools

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, early_stopping
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, RocCurveDisplay
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split,RandomizedSearchCV, StratifiedKFold, cross_val_score

########## BASIC SETUPS

import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 300)
pd.set_option('display.max_columns', 300)
pd.set_option('display.width', 900)

########## LOADING DATASETS

training_data = pd.read_feather('train_data.ftr')
testing_data = pd.read_feather('test_data.ftr')
train_labels = pd.read_csv('train_labels.csv')
sample = pd.read_csv('sample_submission.csv')

########## UNDERSTANDING THE DATASET

#### VISUALIZING THE DATASET COLUMN NAMES

print(training_data.columns)

#### DISPLAYING SAMPLE DATA

print(training_data.head(5))

def data_report(data):
    data_col = []
    data_type = []
    data_uniques = []
    data_n_uniques = []
    
    for i in data.columns:
        data_col.append(i)
        data_type.append(data[i].dtypes)
        data_uniques.append(data[i].unique()[:5])
        data_n_uniques.append(data[i].nunique())
    
    return pd.DataFrame({'Column': data_col,
                         'd_type': data_type,
                         'unique_sample': data_uniques,
                         'n_uniques': data_n_uniques})

# show unique samples of the data

print(data_report(training_data))

def information(a,b):
    a = a
    b = b

    print("\nUNDERSTANDING THE CUSTOMER DATA\n")

    print("\nTRAIN DATA SET INFORMATION\n")
    print(a.info())
    print("TEST DATA SET INFORMATION\n")
    print(b.info())

    print("\nTRAIN DATA SET SHAPE\n")
    print(a.shape)
    print("TEST DATA SET SHAPE\n")
    print(b.shape)
    
    del_var = [c for c in a.columns if c.startswith('D_')]
    spend_var = [c for c in a.columns if c.startswith('S_')]
    pay_var = [c for c in a.columns if c.startswith('P_')]
    bal_var = [c for c in a.columns if c.startswith('B_')]
    risk_var = [c for c in a.columns if c.startswith('R_')]
    if len(del_var) != 0:
        print(f'\nNumber of Delinquency variables: {len(del_var)}')
        print(f'Number of Spend variables: {len(spend_var)}')
        print(f'Number of Payment variables: {len(pay_var)}')
        print(f'Number of Balance variables: {len(bal_var)}')
        print(f'Number of Risk variables: {len(risk_var)}')

    values= [len(del_var), len(spend_var),len(pay_var), len(bal_var),len(risk_var)]

    return values


values = information(training_data, testing_data)

########## DATA CLEANING

columns = training_data.columns[(training_data.isna().sum()/len(training_data))*100>30]

train_data = training_data.drop(columns, axis=1)
test_data = testing_data.drop(columns, axis=1)

values =information(train_data, test_data)

########## EXPLORATORY DATA ANALYSIS

#colors = ['#0b1d78', '#0045a5', '#0069c0', '#008ac5', '#00a9b5', '#00c698', '#1fe074']

labels=['Delinquency', 'Spend','Payment','Balance','Risk']
values = values
fig = px.pie(train_data, values=values, names=labels)
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.update_layout(title= " Variables and Length : AMEX Data")
fig.show()

unique_cx_train = len(train_data.groupby("customer_ID")['customer_ID'].count())
print("\nCount of unique customers in training data: ", unique_cx_train)
unique_cx_test = len(test_data.groupby("customer_ID")['customer_ID'].count())
print("\nCount of unique customers in testing data: ", unique_cx_test)

train_data.groupby("customer_ID").size()

cx_prof = train_data.groupby("customer_ID")['customer_ID'].count().values
cx_test = test_data.groupby("customer_ID")['customer_ID'].count().values

fig = go.Figure()
fig.add_trace(go.Histogram(
    y = cx_prof,
    ybins = dict(size = 0.6),
    marker_color= '#0069c0'))
fig.update_layout(
    template = "plotly_dark",
    title = " Customer profile count : training data",
    yaxis_title = "Number of months",
    bargap = 0.2
)
fig.show()

fig = go.Figure()
fig.add_trace(go.Histogram(
    y = cx_test,
    ybins = dict(size = 0.6),
    marker_color= '#0069c0'))
fig.update_layout(
    template = "plotly_dark",
    title = " Customer profile count : test data",
    yaxis_title = "Number of months"
)
fig.show()

#From here we can see the dsitribution of profile length is common between train and test data.

del cx_prof
del cx_test
gc.collect()

count = train_data.groupby("customer_ID")['customer_ID'].count()
target_data = pd.DataFrame({"customer_ID":count.index, "count": count.values})
target_data = target_data.merge(train_labels, on='customer_ID', how='left')

sns.countplot(data = target_data,
              y='count',hue='target',
              orient='h',
              palette = ['#0069c0','#008ac5']).set(title=" Target's length : training data", xlabel='target')
plt.show()

#nearly 30 - 50 % of all profile length has target 1 (default)
#Can't get a great correlation between profile length and target -- but thinking like keeping this information may help.


del target_data
gc.collect()

temp=dict(layout=go.Layout(font=dict(family="Franklin Gothic", size=12), 
                           height=500, width=1000))

train = train_data.groupby('customer_ID').tail(1).set_index('customer_ID')
print("\nThe training data begins on {} and ends on {}.".format(train['S_2'].min().strftime('%m-%d-%Y'),train['S_2'].max().strftime('%m-%d-%Y')))
print("\nThere are {:,.0f} customers in the training set and {} features.".format(train.shape[0],train.shape[1]))

test = test_data.groupby('customer_ID').tail(1).set_index('customer_ID')
print("\nThe test data begins on {} and ends on {}.".format(test['S_2'].min().strftime('%m-%d-%Y'),test['S_2'].max().strftime('%m-%d-%Y')))
print("\nThere are {:,.0f} customers in the test set and {} features.".format(test.shape[0],test.shape[1]))


#There are 5.5 million rows for training and 11 million rows of test data.


del test['S_2']
gc.collect()

titles=['Delinquency '+str(i).split('_')[1] if i.startswith('D') else 'Spend '+str(i).split('_')[1] 
        if i.startswith('S') else 'Payment '+str(i).split('_')[1]  if i.startswith('P') 
        else 'Balance '+str(i).split('_')[1] if i.startswith('B') else 
        'Risk '+str(i).split('_')[1] for i in train.columns[:-1]]
cat_cols=['Balance 30', 'Balance 38', 'Delinquency 63', 'Delinquency 64', 'Delinquency 66', 'Delinquency 68',
          'Delinquency 114', 'Delinquency 116', 'Delinquency 117', 'Delinquency 120', 'Delinquency 126', 'Target']
test.columns=titles[1:]
titles.append('Target')
train.columns=titles

target=train.Target.value_counts(normalize=True)
target.rename(index={1:'Default',0:'Paid'},inplace=True)
pal, color=['#0069c0', '#008ac5'], ['#0069c0', '#008ac5']


fig=go.Figure()
fig.add_trace(go.Pie(labels=target.index, values=target*100, hole=.45, 
                     showlegend=True,sort=False, 
                     marker=dict(colors=color,line=dict(color=pal,width=2.5)),
                     hovertemplate = "%{label} Accounts: %{value:.2f}%<extra></extra>"))
fig.update_layout(template=temp, title=' Target Distribution', 
                  legend=dict(traceorder='reversed',y=1.05,x=0),
                  uniformtext_minsize=15, uniformtext_mode='hide',width=700)
fig.show()


target=pd.DataFrame(data={'Default':train.groupby('Spend 2')['Target'].mean()*100})
target['Paid']=np.abs(train.groupby('Spend 2')['Target'].mean()-1)*100
rgb=['rgba'+str(matplotlib.colors.to_rgba(i,0.7)) for i in pal]
fig=go.Figure()
fig.add_trace(go.Bar(x=target.index, y=target.Paid, name='Paid',
                     text=target.Paid, texttemplate='%{text:.0f}%', 
                     textposition='inside',insidetextanchor="middle",
                     marker=dict(color=color[0],line=dict(color=pal[0],width=1.5)),
                     hovertemplate = "<b>%{x}</b><br>Paid accounts: %{y:.2f}%"))
fig.add_trace(go.Bar(x=target.index, y=target.Default, name='Default',
                     text=target.Default, texttemplate='%{text:.0f}%', 
                     textposition='inside',insidetextanchor="middle",
                     marker=dict(color=color[1],line=dict(color=pal[1],width=1.5)),
                     hovertemplate = "<b>%{x}</b><br>Default accounts: %{y:.2f}%"))
fig.update_layout(template=temp,title=' Distribution of Default by Day', 
                  barmode='relative', yaxis_ticksuffix='%', width=1400,
                  legend=dict(orientation="h", traceorder="reversed", yanchor="bottom",y=1.1,xanchor="left", x=0))
fig.show()


plot_df=train.reset_index().groupby('Spend 2')['customer_ID'].nunique().reset_index()
fig=go.Figure()
fig.add_trace(go.Scatter(x=plot_df['Spend 2'], 
                         y=plot_df['customer_ID'], mode='lines',
                         line=dict(color=pal[1], width=3), 
                         hovertemplate = ''))
fig.update_layout(template=temp, title=" Frequency of Customer Statements", 
                  hovermode="x unified", width=800,height=500,
                  xaxis_title='Statement Date', yaxis_title='Number of Statements Issued')
fig.show()
del train['Spend 2']

########## DATA PREPROCESSING

train_df =train.groupby('customer_ID').tail(1)
test_df = test.groupby('customer_ID').tail(1)

train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)

information(train_df, test_df)

#getting numerical and categorical column names
num_cols = train_df.select_dtypes([np.int64,np.float16]).columns
cat_cols = ['Delinquency 63', 'Delinquency 64', 'Delinquency 68', 'Balance 30',
            'Balance 38', 'Delinquency 114', 'Delinquency 116', 'Delinquency 117',
            'Delinquency 120', 'Delinquency 126']

#Encoding categorical columns

label_enc = LabelEncoder()
for categorical in cat_cols:
    train_df[categorical] = label_enc.fit_transform(train_df[categorical])
    test_df[categorical] = label_enc.transform(test_df[categorical])

train_df = train_df.replace(np.nan, 0)

########## DATA VISUALIZATION & MODEL BUILDING

#defining X and Y data 
#X,Y=train_df.drop(['customer_ID','Target','Spend 2'],axis=1),train_df['Target']
X,Y=train_df.drop(['Target'],axis=1),train_df['Target']

#making sure we have same columns in test data as in train
col=[c for c in X.columns]
test_x=test_df[col]

del test_df,train,test,train_df
gc.collect()

# getting data ready to build model
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size=0.40,
                                                    random_state=30,
                                                    shuffle=True)


categorical_features_indices=[]
for c in cat_cols:
    a=X_train.columns.get_loc(c)
    categorical_features_indices.append(a)

del X,Y
gc.collect()

print(f"""
X_train shape: {X_train.shape}
X_test shape: {X_test.shape}
y_train shape: {Y_train.shape}
y_test shape: {Y_test.shape}
""")

# Create our imputer to replace missing values with the mean e.g.
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp = imp.fit(X_train)

# Impute our data, then train
X_train = pd.DataFrame(imp.transform(X_train))

def models(X_train,Y_train):

  XGB = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1)
  XGB.fit(X_train, Y_train)  

  LGBM = LGBMClassifier(n_estimators=1000,boosting_type ='gbdt',max_depth=6,
                        learning_rate=0.1,num_leaves= 50,max_bins= 500)
  LGBM.fit(X_train, Y_train)

  logreg = LogisticRegression(solver = 'liblinear',random_state=0)
  logreg.fit(X_train, Y_train)

  random= RandomForestClassifier(n_estimators=200,max_features='sqrt',
                                 bootstrap=True,max_depth=6,min_samples_leaf=1,
                                 min_samples_split=5,n_jobs=-1)
  random.fit(X_train, Y_train)

  #print model accuracy on the training data.
  print('[1]XGB Classifier Accuracy:', XGB.score(X_train, Y_train))
  print('[2]LGBM Accuracy:', LGBM.score(X_train, Y_train))
  print('[3]Logistic regression Accuracy:', logreg.score(X_train, Y_train))
  print('[4]Random forest Accuracy:', random.score(X_train, Y_train))

  return XGB, LGBM, logreg, random

model = models(X_train,Y_train)

for i in range(len(model)):
  print()
  print(model)
  #Check precision, recall, f1-score
  print("\nclassification report:" )
  print( classification_report(Y_test, model[i].predict(X_test)) )
  print("\nconfusion matrix:")
  print(confusion_matrix(Y_test, model[i].predict(X_test)))
  #Another way to get the models accuracy on the test data
  print("\nAccuracy score:")
  print( accuracy_score(Y_test, model[i].predict(X_test)))
  pre_score = precision_score(Y_test, model[i].predict(X_test))
  print("\nprecision: ",pre_score)
  rec_score = recall_score(Y_test, model[i].predict(X_test))
  print("\nrecall: ",rec_score)
  f_score = f1_score(Y_test, model[i].predict(X_test), average = 'weighted')
  print("\nf1_score: ",f_score)
  print()#Print a new line
  
  false, true, thresholds = roc_curve(Y_test, model[i].predict(X_test))
  roc_auc = auc(false, true)
  display = RocCurveDisplay(fpr=false, tpr=true, roc_auc=roc_auc)
  display.plot()
  tot_pos = (Y_test == 1).sum()
  tot_neg = (Y_test == 0).sum() * 20
  fourth = int(0.04 * (tot_pos + tot_neg))
  plt.plot(false, true, color = '#0069c0',lw=3)
  plt.fill_between(false, true, color = '#00a9b5')
  plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
  plt.plot([fourth / tot_neg, 0], [0, fourth / tot_pos],
         color="green", lw=3, linestyle="-") 
  fourth_index = np.argmax((false * tot_neg + true * tot_pos >= fourth))
  plt.scatter([false[fourth_index]],[true[fourth_index]],
              s=100) # intersection of roc curve with four percent line
  plt.gca().set_aspect('equal')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.0])
  plt.xlabel("False Positive Rate")
  plt.ylabel("True Positive Rate (Recall)")
  plt.title("Receiver operating characteristic")
  plt.show()
