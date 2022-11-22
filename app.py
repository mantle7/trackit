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
import streamlit as st
import plotly_express as px
import shap

from trackit_modular import df_preprocessing

st.set_page_config(
	page_title="TrackIt",
	layout='wide',
	page_icon="🤖")

hide_streamlit_style = """
            <style>
            
            footer {visibility: hidden;}
            </style>
            """
#MainMenu {visibility: hidden;}
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title("Welcome to TrackIt")

global df

df=pd.read_csv('Leads X Education.csv')

# In[10]:

original_df_copy=df.copy() # for prospect id and lead number

# In[11]:

df.columns=[x.lower() for x in df.columns]
df.columns=df.columns.str.replace(' ','_')

final_df,optimal_score,cbc,X_test=df_preprocessing(df)

# st.write(final_df)

final_df_idx=final_df.index

original_df_copy=original_df_copy.iloc[final_df_idx,:]

original_df_copy['Scores']=final_df['Scores']
original_df_copy['Scores']=original_df_copy['Scores'].astype(int)

test_df=original_df_copy.copy()

percentile=test_df.sort_values(by=['Scores'],ascending=True)

percentile=percentile.reset_index(drop=True)
percentile_idx=percentile.shape[0]*0.99
percentile_idx=int(percentile_idx)

percentile_score=percentile.iloc[percentile_idx,percentile.shape[1]-1]

# x=percentile.iloc[percentile_idx-1,:]
# percentile_idx_score=percentile.iloc[percentile_idx,test_df.shape[1]-1]

df_no_go=test_df[test_df['Scores']<optimal_score]

df_okay=test_df[test_df['Scores']>=optimal_score]
df_okay=df_okay[df_okay['Scores']<percentile_score]

df_love=test_df[test_df['Scores']>=percentile_score]

donut_test_df=test_df.copy()
win_funnel=[]
for i in range(donut_test_df.shape[0]):
	if donut_test_df.iloc[i,donut_test_df.shape[1]-1]<optimal_score:
		win_funnel.append('Cold Lead')
	elif donut_test_df.iloc[i,donut_test_df.shape[1]-1]>=percentile_score:
		win_funnel.append('Hot Lead')
	else:
		win_funnel.append('Warm Lead')

donut_test_df['Win Funnel']=win_funnel
value1=df_love.shape[0]
value2=df_okay.shape[0]
value3=df_no_go.shape[0]

# st.session_state['my_input4']=donut_test_df

labels=["Hot Lead","Warm Lead","Cold Lead"]
values=[value1,value2,value3]

#DONUT FOR SEGMENTATION ##############################################
fig1=px.pie(donut_test_df,values=values,names=labels,hole=0.55,
	color_discrete_map={'Hot Lead':'blue',
                                 'Warm Lead':'green',
                                 "Cold Lead":'red'
                                 })

col1,col2=st.columns((1.75,3))

with col1:
	col1.metric("Number of Active Leads",test_df.shape[0])

col2.subheader('Lead Segmentation')
with col2:    
    # col2.header = "Leads By Win Funnel"
    st.plotly_chart(fig1)


# explainer=shap.Explainer(cbc)
# shap_values=explainer(X_test)
# fig1=shap.plots.bar(shap_values, max_display=X_test.shape[0])
# st.write(fig1)
# st.write(original_df_copy)

feature_importance = cbc.feature_importances_
sorted_idx = np.argsort(feature_importance)
fig = plt.figure(figsize=(5,5))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center',height=[0.5])
plt.yticks(range(len(sorted_idx)), np.array(X_test.columns)[sorted_idx])
# plt.title('Feature Importance')
ex=st.expander("Click to view Feature Importances")

with ex:
	st.header("Feature Importances")
	st.write(fig)

win = st.selectbox("Filter results on", donut_test_df['Win Funnel'].unique().tolist())

attributes=test_df.columns.tolist()
attr=st.multiselect('Attributes',options=attributes,default=attributes)

df_selection1_idx=donut_test_df[donut_test_df['Win Funnel']==win].index
df_selection2=donut_test_df.loc[df_selection1_idx,attr]
st.dataframe(df_selection2)






