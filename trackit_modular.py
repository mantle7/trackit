
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve

from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
    
from sklearn.model_selection import train_test_split


# In[9]:





# In[47]:


def df_preprocessing(df2):
    df2=df2.drop(['a_free_copy_of_mastering_the_interview','country','get_updates_on_dm_content',
              'i_agree_to_pay_the_amount_through_cheque','specialization',
             'update_me_on_supply_chain_content','what_matters_most_to_you_in_choosing_a_course',
              'x_education_forums'],axis='columns')
    
    df2=df2.drop(['asymmetrique_activity_index','asymmetrique_profile_index'
            ,'asymmetrique_activity_score','asymmetrique_profile_score'],axis='columns')
    
    cols_to_remove=[]
    cols_to_remove.append('lead_quality')
    
    df2['what_is_your_current_occupation']=df2.what_is_your_current_occupation.replace(np.nan,'Other')
    df2.city=df2.city.replace([np.nan,'Select'],'Other Cities')
    
    df2=df2.drop(cols_to_remove,axis='columns')
    
    df2=df2.dropna()
    
    df2=df2.drop(['prospect_id','lead_number'],axis='columns')
    
    df2=df2.replace(['No','Yes'],[0,1])
    
    test_converted_values_df=df2.copy()
    
    Y=df2.converted
    X=df2.drop(['converted'],axis='columns')
    
    numeric_columns=[x for x in X.select_dtypes(include=np.number).columns]
    cat_columns=[x for x in X.select_dtypes(include=np.object).columns]
    
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.65,random_state=10)

    cbc=CatBoostClassifier(iterations=20)  #102  od_wait= 40
    cbc=cbc.fit(X_train,Y_train,cat_features=cat_columns,silent=True)
    
    Y_predicted_test=cbc.predict(X_test)
    Y_predicted_train=cbc.predict(X_train)
    Y_predicted_total=cbc.predict(X)
    
    probs_train=cbc.predict_proba(X_train)
    probs_train=[x[1] for x in probs_train]

    probs_test=cbc.predict_proba(X_test)
    probs_test=[x[1] for x in probs_test]

    probs_total=cbc.predict_proba(X)
    probs_total=[x[1] for x in probs_total]    
    
    fpr,tpr,thresholds=roc_curve(Y,probs_total,drop_intermediate = False )
    
    optimal_idx=np.argmax(tpr-fpr)
    optimal_score=thresholds[optimal_idx]
    
    df2['Scores']=probs_total
    
    df2['Scores']=df2['Scores']*100
    
    df2=df2.drop(['converted'],axis='columns')
    
    return df2,optimal_score,cbc,X_test
    
    


# In[48]:


# final_df,optimal_score,test_converted_values_df=df_preprocessing(df)
