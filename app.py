from pathlib import Path
import hashlib, io, zipfile
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from engine import train
ROOT=Path(__file__).parent
st.set_page_config(page_title='Student Performance Lab',page_icon='🎓',layout='wide')
st.title('🎓 Student Performance Lab')
st.caption('Explore data · Compare models · Explain predictions')
@st.cache_data(ttl=86400)
def real_data():
    import urllib.request
    with urllib.request.urlopen('https://archive.ics.uci.edu/static/public/320/student+performance.zip',timeout=20) as response:
        z=zipfile.ZipFile(io.BytesIO(response.read()))
    if 'student.zip' in z.namelist(): z=zipfile.ZipFile(io.BytesIO(z.read('student.zip')))
    name=next(n for n in z.namelist() if n.endswith('student-mat.csv'))
    return pd.read_csv(z.open(name),sep=';')
mode=st.sidebar.radio('Dataset',['Synthetic demo','UCI real student data','Upload CSV'])
if mode=='Synthetic demo':
    df=pd.read_csv(ROOT/'student_performance.csv'); default='final_score'; st.info('Synthetic demo data: not evidence of real-world accuracy.')
elif mode=='UCI real student data':
    st.markdown('[UCI Student Performance dataset](https://archive.ics.uci.edu/dataset/320/student+performance) · Mathematics · G3 final grade (0–20).')
    try: df=real_data()
    except Exception as e:
        st.warning('Could not fetch UCI data. Download student-mat.csv from UCI and use Upload CSV.'); st.stop()
    default='G3'
else:
    f=st.sidebar.file_uploader('Upload CSV',type='csv')
    if not f: st.info('Upload a CSV to begin.'); st.stop()
    try: df=pd.read_csv(f,sep=None,engine='python')
    except Exception as e: st.error(f'Cannot read CSV: {e}'); st.stop()
    default='G3' if 'G3' in df else None
if len(df)>20000: st.error('Maximum 20,000 rows for this demo.'); st.stop()
nums=df.select_dtypes(include='number').columns.tolist()
if not nums: st.error('A numeric target is required.'); st.stop()
target=st.sidebar.selectbox('Target',nums,index=nums.index(default) if default in nums else len(nums)-1)
options=[c for c in df if c!=target]
features=st.sidebar.multiselect('Input features',options,default=[c for c in options if c not in ['G1','G2']])
if target=='G3': st.caption('G1 and G2 are excluded by default for early prediction; include them only if available at prediction time.')
st.sidebar.caption('Exclude identifiers and information unavailable at prediction time to avoid leakage.')
size=st.sidebar.slider('Test fraction',.15,.4,.2,.05)
key=hashlib.sha256(pd.util.hash_pandas_object(df,index=True).values.tobytes()+repr((mode,target,features,size,list(df.columns))).encode()).hexdigest()
a,b,c=st.columns(3); a.metric('Records',len(df)); b.metric('Features',len(features)); c.metric('Missing cells',int(df.isna().sum().sum()))
explore,compare,predict=st.tabs(['Data explorer','Model comparison','Prediction'])
with explore:
    st.dataframe(df.head(100),use_container_width=True)
    st.plotly_chart(px.histogram(df,x=target,title='Target distribution'),use_container_width=True)
    corr=df[nums[:30]].corr(); st.plotly_chart(go.Figure(go.Heatmap(z=corr.values,x=corr.columns,y=corr.index,zmin=-1,zmax=1,colorscale='RdBu')),use_container_width=True)
with compare:
    if st.button('Train and compare models',type='primary'):
        if not features: st.error('Select at least one feature.')
        else:
            try:
                with st.spinner('Running five-fold cross-validation and held-out evaluation…'): st.session_state['fit']=(key,train(df,target,features,size))
            except Exception as e: st.error(str(e))
    fit=st.session_state.get('fit')
    if fit and fit[0]==key:
        results,best,model,points,importance,X=fit[1]
        st.success(f'Best model by training cross-validation MAE: {best}')
        st.dataframe(results.round(3),use_container_width=True,hide_index=True)
        st.plotly_chart(px.bar(results,x='Model',y='CV MAE',error_y='CV MAE SD',title='Cross-validation error (lower is better)'),use_container_width=True)
        fig=px.scatter(points,x='Actual',y='Predicted',title='Held-out actual vs predicted'); lo=points.min().min(); hi=points.max().max(); fig.add_shape(type='line',x0=lo,y0=lo,x1=hi,y1=hi,line_dash='dash'); st.plotly_chart(fig,use_container_width=True)
        st.plotly_chart(px.bar(importance,x='Importance',y='Feature',orientation='h',title='Permutation importance (increase in MAE)'),use_container_width=True)
        st.caption('Importance is predictive association, not causation. Test scores are not used to select the model.')
        st.download_button('Download model comparison',results.to_csv(index=False),'model_comparison.csv')
    else: st.info('Train models for the current dataset and feature selection.')
with predict:
    if not fit or fit[0]!=key: st.info('Train models first.'); st.stop()
    results,best,model,points,importance,X=fit[1]
    with st.form('prediction_'+key):
        row={}
        for col in features:
            if pd.api.types.is_numeric_dtype(X[col]): row[col]=st.number_input(col,value=float(X[col].median()))
            else: row[col]=st.selectbox(col,sorted(X[col].dropna().unique().tolist()))
        submit=st.form_submit_button('Predict',type='primary')
    if submit:
        value=float(model.predict(pd.DataFrame([row]))[0]); st.session_state['prediction']=(key,row,value)
    saved=st.session_state.get('prediction')
    if saved and saved[0]==key:
        _,row,value=saved; st.metric('Estimated '+target,f'{value:.2f}'); st.caption(f'Model: {best}. Estimate only; may fall outside the observed grade range.')
        if 'study_hours' in row and row['study_hours']<X['study_hours'].median(): st.write('Study time is below the dataset median. Consider a consistent study schedule.')
        if 'attendance_percent' in row and row['attendance_percent']<X['attendance_percent'].median(): st.write('Attendance is below the dataset median. Consider attending more consistently.')
        st.caption('Suggestions are descriptive, not proven interventions.')
        st.download_button('Download prediction',pd.DataFrame([{**row,'prediction':value,'model':best}]).to_csv(index=False),'prediction.csv')
