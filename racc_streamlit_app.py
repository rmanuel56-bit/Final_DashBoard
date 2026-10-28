import io
import re
import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import silhouette_score, accuracy_score, precision_score, recall_score, f1_score, classification_report, ConfusionMatrixDisplay

RANDOM_STATE = 42
REQUIRED = ['DATE','TIME','LOCATION','WEATHER','ROAD CONDITION','VEHICLE TYPE','CAUSE','SEVERITY','GENDER']
st.set_page_config(page_title='Bunawan Road Accident Analytics',layout='wide')
st.title('Bunawan Road Accident Analytics & Decision Support')
st.caption('Historical records and screening only; results are not engineering warrants or causal findings.')

def clean_text(x):
    if pd.isna(x): return np.nan
    s=re.sub(r'\s+',' ',str(x)).strip()
    return np.nan if s.lower() in {'','nan','none','null','n/a','na','-'} else s

def norm_loc(x):
    if pd.isna(x): return np.nan
    return re.sub(r'(?i)\bP\s*-?\s*(\d+)\b',lambda m:'P'+m.group(1),str(x)).title().strip()

def parse_date(x):
    if pd.isna(x): return pd.NaT
    s=str(x).strip().replace('.','/').replace('-','/')
    a=pd.to_datetime(s,errors='coerce',dayfirst=True)
    return a if pd.notna(a) else pd.to_datetime(s,errors='coerce',dayfirst=False)

def parse_hour(x):
    if pd.isna(x): return np.nan
    s=str(x).strip()
    for fmt in ('%H:%M:%S','%H:%M','%I:%M %p','%I:%M:%S %p'):
        try: return pd.to_datetime(s,format=fmt).hour
        except (ValueError,TypeError): pass
    return np.nan

def csv_button(label,frame,name):
    st.download_button(label,frame.to_csv(index=False).encode('utf-8-sig'),file_name=name,mime='text/csv')

def plot_counts(frame,column,title):
    counts=frame[column].fillna('Unknown').value_counts().head(12).sort_values()
    fig,ax=plt.subplots(figsize=(9,max(3,len(counts)*0.38)))
    counts.plot(kind='barh',ax=ax)
    ax.set_title(title)
    fig.tight_layout();st.pyplot(fig);plt.close(fig)

def prep_for(cat,num):
    return ColumnTransformer([
        ('cat',OneHotEncoder(handle_unknown='ignore'),cat),
        ('num',Pipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num)
    ])

upload=st.file_uploader('Upload accident records CSV',type='csv')
if upload is None:
    st.info('Upload a CSV to begin. No analysis runs until a valid dataset is uploaded.')
    st.stop()
try:
    raw=pd.read_csv(upload,encoding='utf-8-sig')
except Exception as e:
    st.error(f'Cannot read CSV: {e}');st.stop()
df=raw.copy()
df.columns=[str(c).strip().upper().replace('_',' ') for c in df.columns]
missing=[c for c in REQUIRED if c not in df.columns]
if missing:
    st.error(f'Missing required columns: {missing}');st.write('Available columns:',list(df.columns));st.stop()
if df.empty:
    st.warning('The uploaded file has no records.');st.stop()

with st.expander('1. Upload and source audit',expanded=True):
    a,b,c=st.columns(3)
    a.metric('Source records',len(raw));b.metric('Columns',len(raw.columns));c.metric('Exact duplicates',int(raw.duplicated().sum()))
    st.dataframe(raw.head(20),use_container_width=True)

with st.expander('2. Data cleaning and preprocessing',expanded=True):
    for c in REQUIRED:
        df[c+' RAW']=df[c]
        df[c]=df[c].map(clean_text)
    for col,mapping in {
        'WEATHER':{'sunny':'Clear/Sunny','clear':'Clear/Sunny','rainy':'Rain/Rainy'},
        'ROAD CONDITION':{'wet':'Wet','clear':'Dry','dry':'Dry'},
        'SEVERITY':{'minor injry':'Minor Injury','major injury':'Major Injury'},
        'VEHICLE TYPE':{'motorcyle':'Motorcycle','motor cycle':'Motorcycle'}
    }.items():
        df[col]=df[col].map(lambda v:mapping.get(v.lower(),v) if isinstance(v,str) else v)
    df['DATE PARSED']=df['DATE'].map(parse_date)
    df['HOUR']=df['TIME'].map(parse_hour)
    df['MONTH']=df['DATE PARSED'].dt.month
    df['YEAR']=df['DATE PARSED'].dt.year
    df['DAY OF WEEK']=df['DATE PARSED'].dt.day_name()
    df['WEEKEND']=df['DATE PARSED'].dt.dayofweek.ge(5).astype(float)
    df.loc[df['DATE PARSED'].isna(),'WEEKEND']=np.nan
    df['LOCATION CLEAN']=df['LOCATION'].map(norm_loc)
    a,b=st.columns(2)
    a.metric('Unparseable dates',int((df['DATE'].notna() & df['DATE PARSED'].isna()).sum()))
    b.metric('Unparseable times',int((df['TIME'].notna() & df['HOUR'].isna()).sum()))
    st.dataframe(df[REQUIRED].isna().sum().rename('Missing values').to_frame(),use_container_width=True)
    st.dataframe(df.head(20),use_container_width=True)
    csv_button('Download cleaned dataset',df,'RACC_cleaned_audit.csv')

years=sorted(df['YEAR'].dropna().astype(int).unique().tolist())
locations=sorted(df['LOCATION CLEAN'].dropna().astype(str).unique().tolist())
st.subheader('Filters')
a,b=st.columns(2)
selected_years=a.multiselect('Years',years,default=years)
selected_locations=b.multiselect('Locations',locations,default=locations)
view=df.copy()
if years: view=view[view['YEAR'].isin(selected_years)]
if locations: view=view[view['LOCATION CLEAN'].isin(selected_locations)]
view=view.copy()
if view.empty:
    st.warning('No records match the selected filters.');st.stop()

with st.expander('3. Exploratory data analysis',expanded=True):
    a,b,c=st.columns(3)
    a.metric('Filtered records',len(view));b.metric('Locations',view['LOCATION CLEAN'].nunique());c.metric('Fatal records',int(view['SEVERITY'].fillna('').str.lower().eq('fatal').sum()))
    for col,title in [('SEVERITY','Severity'),('LOCATION CLEAN','Locations'),('CAUSE','Recorded causes'),('ROAD CONDITION','Road conditions')]:
        plot_counts(view,col,title)
    dated=view.dropna(subset=['DATE PARSED'])
    if not dated.empty:
        monthly=dated.groupby(dated['DATE PARSED'].dt.to_period('M')).size()
        monthly.index=monthly.index.astype(str)
        st.subheader('Monthly accident records');st.line_chart(monthly)
    hours=view['HOUR'].dropna().astype(int).value_counts().reindex(range(24),fill_value=0).sort_index()
    st.subheader('Records by hour');st.bar_chart(hours)
    for col in ['WEATHER','ROAD CONDITION','VEHICLE TYPE']:
        st.write(f'{col} versus severity')
        st.dataframe(pd.crosstab(view[col].fillna('Unknown'),view['SEVERITY'].fillna('Unknown')))

catc=['WEATHER','ROAD CONDITION','VEHICLE TYPE','CAUSE','LOCATION CLEAN','DAY OF WEEK']
numc=['HOUR','MONTH']
with st.expander('4. K-Means clustering'):
    if len(view)<4:
        st.info('At least four records are needed for this clustering analysis.')
    else:
        try:
            cx=view[catc+numc].copy()
            for c in catc: cx[c]=cx[c].fillna('Unknown').astype(str)
            for c in numc: cx[c]=pd.to_numeric(cx[c],errors='coerce')
            if cx[numc].notna().sum().min()==0: raise ValueError('Hour or month is entirely missing.')
            Xc=prep_for(catc,numc).fit_transform(cx)
            silrows=[]
            for k in range(2,min(8,len(view)-1)+1):
                labels=KMeans(n_clusters=k,random_state=RANDOM_STATE,n_init=10).fit_predict(Xc)
                if len(set(labels))<2 or len(set(labels))>=len(view): continue
                score=silhouette_score(Xc,labels,sample_size=min(1000,len(view)),random_state=RANDOM_STATE)
                silrows.append({'k':k,'silhouette':score})
            sil_df=pd.DataFrame(silrows)
            if sil_df.empty: st.info('No valid cluster count could be evaluated.')
            else:
                st.dataframe(sil_df)
                best_k=int(sil_df.loc[sil_df['silhouette'].idxmax(),'k'])
                view['KMEANS CLUSTER']=KMeans(n_clusters=best_k,random_state=RANDOM_STATE,n_init=10).fit_predict(Xc)
                st.metric('Selected clusters',best_k)
                for col in ['LOCATION CLEAN','CAUSE','SEVERITY','WEATHER','ROAD CONDITION']:
                    st.write(f'Cluster versus {col}');st.dataframe(pd.crosstab(view['KMEANS CLUSTER'],view[col].fillna('Unknown')))
        except (ValueError,TypeError) as e: st.warning(f'Clustering skipped: {e}')

with st.expander('5. Association rule mining'):
    assoc_cols=['WEATHER','ROAD CONDITION','VEHICLE TYPE','CAUSE','SEVERITY']
    transactions=[[f'{c}={row[c]}' for c in assoc_cols] for _,row in view[assoc_cols].fillna('Unknown').astype(str).iterrows()]
    te=TransactionEncoder()
    basket=pd.DataFrame(te.fit(transactions).transform(transactions),columns=te.columns_)
    min_support=max(0.02,2/len(basket))
    freq=apriori(basket,min_support=min_support,use_colnames=True,max_len=3) if min_support<=1 else pd.DataFrame()
    if freq.empty: st.info('No frequent itemsets at this support threshold.')
    else:
        try:
            rules=association_rules(freq,metric='confidence',min_threshold=0.25)
            if rules.empty: st.info('No association rules meet the confidence threshold.')
            else:
                rules=rules.copy()
                for col in ['antecedents','consequents']:
                    rules[col]=rules[col].map(lambda x:', '.join(sorted(x)))
                st.dataframe(rules.sort_values(['lift','confidence'],ascending=False)[['antecedents','consequents','support','confidence','lift']].head(30),use_container_width=True)
        except ValueError as e: st.info(f'No valid rules: {e}')

with st.expander('6. Anomaly detection'):
    if len(view)<20: st.info('At least 20 records required.')
    else:
        try:
            ac=view[catc[:-1]+numc].copy()
            for c in catc[:-1]: ac[c]=ac[c].fillna('Unknown').astype(str)
            if ac[numc].notna().sum().min()==0: raise ValueError('Hour or month is entirely missing.')
            Xa=prep_for(catc[:-1],numc).fit_transform(ac)
            iso=IsolationForest(contamination=0.03,random_state=RANDOM_STATE)
            view['ANOMALY FLAG']=np.where(iso.fit_predict(Xa)==-1,'Review unusual record','Typical pattern')
            st.dataframe(view.loc[view['ANOMALY FLAG']=='Review unusual record',['DATE','TIME','LOCATION CLEAN','WEATHER','ROAD CONDITION','VEHICLE TYPE','CAUSE','SEVERITY']].head(30))
            st.caption('Anomaly flags indicate unusual combinations, not confirmed errors.')
        except ValueError as e: st.warning(f'Anomaly detection skipped: {e}')

with st.expander('7–8. Classification and cross-validation'):
    md=view.dropna(subset=['SEVERITY']).copy()
    counts=md['SEVERITY'].value_counts()
    md=md[md['SEVERITY'].isin(counts[counts>=2].index)].copy()
    features=['WEATHER','ROAD CONDITION','VEHICLE TYPE','CAUSE','LOCATION CLEAN','HOUR','MONTH','DAY OF WEEK','WEEKEND']
    catf=['WEATHER','ROAD CONDITION','VEHICLE TYPE','CAUSE','LOCATION CLEAN','DAY OF WEEK']
    numf=['HOUR','MONTH','WEEKEND']
    if len(md)<8 or md['SEVERITY'].nunique()<2:
        st.warning('Not enough labeled records and severity classes for classification.')
    else:
        X=md[features].copy();y=md['SEVERITY']
        for c in catf: X[c]=X[c].fillna('Unknown').astype(str)
        if X[numf].notna().sum().min()==0:
            st.warning('A numeric model feature has no usable values; classification skipped.')
        else:
            test_n=max(int(np.ceil(len(y)*0.25)),y.nunique())
            if test_n>=len(y) or len(y)-test_n<y.nunique() or y.value_counts().min()<2:
                st.warning('Insufficient class support for a stratified train/test split.')
            else:
                Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=test_n,random_state=RANDOM_STATE,stratify=y)
                estimators={
                    'Decision Tree':DecisionTreeClassifier(random_state=RANDOM_STATE,class_weight='balanced',max_depth=6,min_samples_leaf=5,min_samples_split=10),
                    'Random Forest':RandomForestClassifier(n_estimators=250,random_state=RANDOM_STATE,class_weight='balanced_subsample',min_samples_leaf=2,n_jobs=-1)
                }
                metric_rows=[]
                for name,est in estimators.items():
                    pipe=Pipeline([('prep',prep_for(catf,numf)),('model',est)])
                    pipe.fit(Xtr,ytr);pred=pipe.predict(Xte)
                    metric_rows.append({'Model':name,'Accuracy':accuracy_score(yte,pred),'Precision macro':precision_score(yte,pred,average='macro',zero_division=0),'Recall macro':recall_score(yte,pred,average='macro',zero_division=0),'F1 macro':f1_score(yte,pred,average='macro',zero_division=0),'F1 weighted':f1_score(yte,pred,average='weighted',zero_division=0)})
                    st.subheader(name);st.text(classification_report(yte,pred,zero_division=0))
                    fig,ax=plt.subplots(figsize=(7,5));ConfusionMatrixDisplay.from_predictions(yte,pred,ax=ax,xticks_rotation=45);fig.tight_layout();st.pyplot(fig);plt.close(fig)
                metrics_df=pd.DataFrame(metric_rows)
                st.dataframe(metrics_df,use_container_width=True)
                csv_button('Download model comparison',metrics_df,'Bunawan_Model_Comparison.csv')
                st.write('Highest test macro F1:',metrics_df.sort_values('F1 macro',ascending=False).iloc[0]['Model'])
                folds=min(5,int(ytr.value_counts().min()))
                if folds>=2:
                    cvrows=[]
                    for name,est in estimators.items():
                        scores=cross_validate(Pipeline([('prep',prep_for(catf,numf)),('model',est)]),Xtr,ytr,cv=StratifiedKFold(n_splits=folds,shuffle=True,random_state=RANDOM_STATE),scoring=['accuracy','precision_macro','recall_macro','f1_macro'])
                        row={'Model':name}
                        for k,v in scores.items():
                            if k.startswith('test_'): row[k+'_mean']=np.mean(v);row[k+'_std']=np.std(v)
                        cvrows.append(row)
                    cv_df=pd.DataFrame(cvrows)
                    st.dataframe(cv_df,use_container_width=True)
                    csv_button('Download cross-validation',cv_df,'Bunawan_Cross_Validation.csv')
                else: st.info('Cross-validation skipped: fewer than two training examples in a class.')

with st.expander('9. Road inspection priority',expanded=True):
    p=view.copy()
    sev=p['SEVERITY'].fillna('').str.lower()
    p['_fatal']=sev.eq('fatal').astype(int)
    p['_severe']=sev.str.contains('fatal|major injury|serious injury|severe injury',regex=True).astype(int)
    p['_night']=((p['HOUR']>=18)|(p['HOUR']<6)).astype(int)
    p['LOCATION CLEAN']=p['LOCATION CLEAN'].fillna('Unknown')
    priority=p.groupby('LOCATION CLEAN',dropna=False).agg(records=('LOCATION CLEAN','size'),fatal_records=('_fatal','sum'),severe_records=('_severe','sum'),night_records=('_night','sum')).reset_index()
    for col in ['CAUSE','VEHICLE TYPE','WEATHER','ROAD CONDITION']:
        modes=p.groupby('LOCATION CLEAN')[col].agg(lambda s:s.mode().iloc[0] if not s.mode().empty else 'Unknown').reset_index(name='top_'+col.lower().replace(' ','_'))
        priority=priority.merge(modes,on='LOCATION CLEAN',how='left')
    priority['screening_score']=100*(.5*priority['records']/max(priority['records'].max(),1)+.3*priority['severe_records']/max(priority['severe_records'].max(),1)+.2*priority['fatal_records']/max(priority['fatal_records'].max(),1))
    priority['inspection_tier']=pd.cut(priority['screening_score'],[-.01,25,50,75,100.01],labels=['Routine review','Monitor','High-priority inspection','Urgent engineering review'])
    priority=priority.sort_values(['screening_score','records'],ascending=False)
    st.dataframe(priority,use_container_width=True)
    st.bar_chart(priority.head(15).set_index('LOCATION CLEAN')['screening_score'])
    csv_button('Download inspection priorities',priority,'Bunawan_RoadSign_Inspection_Priority.csv')
    st.warning('Screening scores are relative to the currently filtered records, not risk probabilities. Site visits and engineering review are required.')

with st.expander('10. Accident hotspot mapping'):
    st.write('Upload a verified location lookup CSV with LOCATION CLEAN, LATITUDE, LONGITUDE. Do not present approximate coordinates as exact crash sites.')
    geo_upload=st.file_uploader('Location coordinates CSV',type='csv',key='geo')
    if geo_upload is not None:
        try:
            geo=pd.read_csv(geo_upload)
            geo.columns=[str(c).strip().upper().replace('_',' ') for c in geo.columns]
            needed={'LOCATION CLEAN','LATITUDE','LONGITUDE'}
            if not needed.issubset(geo.columns):
                st.error(f'Missing coordinate columns: {sorted(needed-set(geo.columns))}')
            else:
                geo['LOCATION CLEAN']=geo['LOCATION CLEAN'].map(norm_loc)
                for c in ['LATITUDE','LONGITUDE']: geo[c]=pd.to_numeric(geo[c],errors='coerce')
                geo=geo.dropna(subset=['LOCATION CLEAN','LATITUDE','LONGITUDE'])
                geo=geo[geo['LATITUDE'].between(-90,90)&geo['LONGITUDE'].between(-180,180)]
                duplicate_locs=geo['LOCATION CLEAN'].duplicated(keep=False)
                if duplicate_locs.any(): st.warning('Repeated location names in lookup: first coordinate per location used. Verify before interpretation.')
                geo=geo.drop_duplicates('LOCATION CLEAN')
                mapped=priority.merge(geo,on='LOCATION CLEAN',how='inner')
                st.write(f'Matched {len(mapped)} of {len(priority)} locations.')
                if mapped.empty: st.info('No matching locations; verify spelling.')
                else:
                    m=folium.Map(location=[mapped['LATITUDE'].mean(),mapped['LONGITUDE'].mean()],zoom_start=12)
                    HeatMap(mapped[['LATITUDE','LONGITUDE','records']].values.tolist(),radius=18,blur=15).add_to(m)
                    for _,r in mapped.iterrows():
                        popup=f"Records: {r['records']} | Severe: {r['severe_records']} | Score: {r['screening_score']:.1f}"
                        folium.CircleMarker([r['LATITUDE'],r['LONGITUDE']],radius=max(4,min(14,3+math.sqrt(r['records']))),tooltip=str(r['LOCATION CLEAN']),popup=popup,fill=True).add_to(m)
                    st_folium(m,width=None,height=520,returned_objects=[])
                    st.download_button('Download hotspot map HTML',m.get_root().render(),file_name='Bunawan_RoadAccident_Hotspots.html',mime='text/html')
        except Exception as e: st.error(f'Could not process coordinates: {e}')
