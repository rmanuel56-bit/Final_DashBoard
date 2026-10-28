# ============================================================
# BUNAWAN ROAD ACCIDENT DATA MINING AND DASHBOARD
# Data Cleaning | EDA | K-Means | Association Rules
# Anomaly Detection | Decision Tree | Random Forest
# Model Evaluation | Road Inspection Priority | Mapping
# Streamlit Dashboard
# ============================================================


import os
import re
import math
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium

from folium.plugins import HeatMap
from google.colab import files

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    silhouette_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

RANDOM_STATE = 42

pd.set_option('display.max_columns', 100)
sns.set_theme()

# ============================================================
# 1. UPLOAD AND LOAD DATASET
# ============================================================

print("=" * 60)
print("1. UPLOAD ROAD ACCIDENT DATASET")
print("=" * 60)

uploaded = files.upload()

csv_name = next(
    (n for n in uploaded if n.lower().endswith('.csv')),
    None
)

if not csv_name:
    raise ValueError("Please upload your RACC CSV file.")

raw = pd.read_csv(csv_name, encoding='utf-8-sig')
df = raw.copy()

df.columns = [
    str(c).strip().upper().replace('_', ' ')
    for c in df.columns
]

required = [
    'DATE', 'TIME', 'LOCATION', 'WEATHER',
    'ROAD CONDITION', 'VEHICLE TYPE',
    'CAUSE', 'SEVERITY', 'GENDER'
]

missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(
        f"Missing columns: {missing}. "
        f"Available columns: {df.columns.tolist()}"
    )

print("Total records:", len(df))
print("Total columns:", len(raw.columns))
print("Exact duplicate source rows:", int(raw.duplicated().sum()))

display(df.head())

# ============================================================
# 2. DATA CLEANING AND PREPROCESSING
# ============================================================

print("\n" + "=" * 60)
print("2. DATA CLEANING AND PREPROCESSING")
print("=" * 60)


def clean_text(x):
    if pd.isna(x):
        return np.nan

    x = re.sub(r'\s+', ' ', str(x)).strip()

    if x.lower() in {
        '', 'nan', 'none', 'null', 'n/a', 'na', '-'
    }:
        return np.nan

    return x


for c in required:
    df[c + ' RAW'] = df[c]
    df[c] = df[c].map(clean_text)


def parse_date(v):
    if pd.isna(v):
        return pd.NaT

    s = re.sub(
        r'/+',
        '/',
        str(v).strip().replace('.', '/').replace('-', '/')
    )

    d = pd.to_datetime(s, errors='coerce', dayfirst=True)

    if pd.notna(d):
        return d

    return pd.to_datetime(s, errors='coerce', dayfirst=False)


df['DATE PARSED'] = df['DATE'].map(parse_date)

try:
    df['TIME PARSED'] = pd.to_datetime(
        df['TIME'],
        errors='coerce',
        format='mixed'
    )
except (TypeError, ValueError):
    df['TIME PARSED'] = pd.to_datetime(
        df['TIME'],
        errors='coerce'
    )

bad = df['TIME'].notna() & df['TIME PARSED'].isna()

if bad.any():
    df.loc[bad, 'TIME PARSED'] = pd.to_datetime(
        df.loc[bad, 'TIME'],
        errors='coerce'
    )

df['HOUR'] = df['TIME PARSED'].dt.hour
df['MONTH'] = df['DATE PARSED'].dt.month
df['YEAR'] = df['DATE PARSED'].dt.year
df['DAY OF WEEK'] = df['DATE PARSED'].dt.day_name()

df['WEEKEND'] = (
    df['DATE PARSED'].dt.dayofweek.ge(5).astype(float)
)

df.loc[df['DATE PARSED'].isna(), 'WEEKEND'] = np.nan

# Standardize selected category labels
df['WEATHER'] = df['WEATHER'].replace({
    'Sunny': 'Clear/Sunny',
    'Clear': 'Clear/Sunny',
    'Rainy': 'Rain/Rainy'
})

df['ROAD CONDITION'] = df['ROAD CONDITION'].replace({
    'wet': 'Wet',
    'Clear': 'Dry'
})

df['SEVERITY'] = df['SEVERITY'].replace({
    'Minor Injry': 'Minor Injury',
    'major Injury': 'Major Injury'
})

df['VEHICLE TYPE'] = df['VEHICLE TYPE'].replace({
    'Motorcyle': 'Motorcycle',
    'Motor Cycle': 'Motorcycle'
})


def normalize_location(x):
    if pd.isna(x):
        return np.nan

    x = re.sub(r'\s+', ' ', str(x)).strip()

    x = re.sub(
        r'(?i)\bP\s*-\s*(\d+)\b',
        r'P\1',
        x
    )

    x = re.sub(
        r'(?i)\bP\s*(\d+)\b',
        lambda m: 'P' + m.group(1),
        x
    )

    return x.title()


df['LOCATION CLEAN'] = df['LOCATION'].map(normalize_location)

print(
    "Unparseable dates:",
    int((df['DATE'].notna() & df['DATE PARSED'].isna()).sum())
)

print(
    "Unparseable times:",
    int((df['TIME'].notna() & df['TIME PARSED'].isna()).sum())
)

print("\nMissing values in original required columns:")
display(
    df[required]
    .isna()
    .sum()
    .rename('Missing Values')
    .to_frame()
)

if df['DATE PARSED'].notna().any():
    print(
        "Date range:",
        df['DATE PARSED'].min(),
        "to",
        df['DATE PARSED'].max()
    )

df.to_csv('RACC_cleaned_audit.csv', index=False)

print("Cleaned dataset saved: RACC_cleaned_audit.csv")

# ============================================================
# 3. FILTER DATA AND EXPLORATORY DATA ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("3. EXPLORATORY DATA ANALYSIS")
print("=" * 60)

years = sorted(
    df['YEAR'].dropna().astype(int).unique().tolist()
)

locs = sorted(
    df['LOCATION CLEAN'].dropna().unique().tolist()
)

selected_years = years
selected_locations = locs

view = df.copy()

if selected_years:
    view = view[view['YEAR'].isin(selected_years)]

if selected_locations:
    view = view[
        view['LOCATION CLEAN'].isin(selected_locations)
    ]

view = view.copy()

if view.empty:
    raise ValueError("No records available after filtering.")

print("Filtered records:", len(view))
print("Unique locations:", view['LOCATION CLEAN'].nunique())

print("\nSeverity distribution:")

display(
    view['SEVERITY']
    .fillna('Unknown')
    .value_counts()
    .to_frame('Records')
)

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

view['SEVERITY'].fillna('Unknown').value_counts().head(12).sort_values().plot(
    kind='barh',
    ax=axes[0, 0]
)
axes[0, 0].set_title('Severity Labels')

view['LOCATION CLEAN'].fillna('Unknown').value_counts().head(12).sort_values().plot(
    kind='barh',
    ax=axes[0, 1]
)
axes[0, 1].set_title('Most Frequent Locations')

view['CAUSE'].fillna('Unknown').value_counts().head(12).sort_values().plot(
    kind='barh',
    ax=axes[1, 0]
)
axes[1, 0].set_title('Most Frequently Entered Causes')

view['ROAD CONDITION'].fillna('Unknown').value_counts().plot(
    kind='bar',
    ax=axes[1, 1]
)
axes[1, 1].set_title('Road Condition')
axes[1, 1].tick_params(axis='x', rotation=35)

plt.tight_layout()
plt.show()

# Monthly accident records
if view['DATE PARSED'].notna().any():

    dated = view.dropna(subset=['DATE PARSED']).copy()

    monthly = dated.groupby(
        dated['DATE PARSED'].dt.to_period('M')
    ).size()

    monthly.index = monthly.index.astype(str)

    plt.figure(figsize=(14, 4))
    monthly.plot(marker='o')
    plt.title('Monthly Accident Records')
    plt.xlabel('Month')
    plt.ylabel('Number of Records')
    plt.xticks(rotation=70)
    plt.tight_layout()
    plt.show()

# Records by hour
hours = (
    view['HOUR']
    .dropna()
    .astype(int)
    .value_counts()
    .sort_index()
    .reindex(range(24), fill_value=0)
)

plt.figure(figsize=(10, 4))
hours.plot(kind='bar')
plt.title('Accident Records by Hour')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Records')
plt.tight_layout()
plt.show()

# Contributing factors versus severity
for c in ['WEATHER', 'ROAD CONDITION', 'VEHICLE TYPE']:
    display(
        pd.crosstab(
            view[c].fillna('Unknown'),
            view['SEVERITY'].fillna('Unknown')
        )
    )

# ============================================================
# 4. K-MEANS CLUSTERING
# ============================================================

print("\n" + "=" * 60)
print("4. K-MEANS CLUSTERING")
print("=" * 60)

catc = [
    'WEATHER',
    'ROAD CONDITION',
    'VEHICLE TYPE',
    'CAUSE',
    'LOCATION CLEAN',
    'DAY OF WEEK'
]

numc = ['HOUR', 'MONTH']

cx = view[catc + numc].copy()

for c in catc:
    cx[c] = cx[c].fillna('Unknown').astype(str)

for c in numc:
    cx[c] = pd.to_numeric(cx[c], errors='coerce')

cprep = ColumnTransformer([
    (
        'cat',
        Pipeline([
            ('imp', SimpleImputer(strategy='most_frequent')),
            ('oh', OneHotEncoder(handle_unknown='ignore'))
        ]),
        catc
    ),
    (
        'num',
        Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('scale', StandardScaler())
        ]),
        numc
    )
])

Xc = cprep.fit_transform(cx)

silrows = []

for k in range(2, min(8, len(view) - 1) + 1):

    kmtry = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=10
    )

    lab = kmtry.fit_predict(Xc)

    try:
        sil = silhouette_score(
            Xc,
            lab,
            sample_size=min(1000, len(view)),
            random_state=RANDOM_STATE
        )
    except Exception:
        sil = np.nan

    silrows.append({
        'k': k,
        'silhouette': sil,
        'inertia': kmtry.inertia_
    })

sil_df = pd.DataFrame(silrows)
display(sil_df)

if not sil_df.empty and sil_df['silhouette'].notna().any():

    best_k = int(
        sil_df.loc[sil_df['silhouette'].idxmax(), 'k']
    )

    km = KMeans(
        n_clusters=best_k,
        random_state=RANDOM_STATE,
        n_init=10
    )

    view['KMEANS CLUSTER'] = km.fit_predict(Xc)

    print("Selected number of clusters:", best_k)

    for c in [
        'LOCATION CLEAN',
        'CAUSE',
        'SEVERITY',
        'WEATHER',
        'ROAD CONDITION'
    ]:
        display(
            pd.crosstab(
                view['KMEANS CLUSTER'],
                view[c].fillna('Unknown')
            )
        )

else:
    view['KMEANS CLUSTER'] = -1
    print("Unable to select clusters using silhouette score.")

# ============================================================
# 5. ASSOCIATION RULE MINING
# ============================================================

print("\n" + "=" * 60)
print("5. ASSOCIATION RULE MINING")
print("=" * 60)

assoc_cols = [
    'WEATHER',
    'ROAD CONDITION',
    'VEHICLE TYPE',
    'CAUSE',
    'SEVERITY'
]

transactions = []

for _, row in view[assoc_cols].fillna('Unknown').astype(str).iterrows():
    transactions.append([
        f'{c}={row[c]}'
        for c in assoc_cols
    ])

te = TransactionEncoder()

basket = pd.DataFrame(
    te.fit(transactions).transform(transactions),
    columns=te.columns_
)

freq = apriori(
    basket,
    min_support=max(0.02, 2 / max(len(basket), 1)),
    use_colnames=True
)

if len(freq):

    rules = association_rules(
        freq,
        metric='confidence',
        min_threshold=0.25
    )

    rules['antecedents'] = rules['antecedents'].apply(
        lambda x: ', '.join(sorted(x))
    )

    rules['consequents'] = rules['consequents'].apply(
        lambda x: ', '.join(sorted(x))
    )

    display(
        rules.sort_values(
            ['lift', 'confidence'],
            ascending=False
        )[
            [
                'antecedents',
                'consequents',
                'support',
                'confidence',
                'lift'
            ]
        ].head(20)
    )

else:
    print("No frequent combinations at this threshold.")

# ============================================================
# 6. ANOMALY DETECTION
# ============================================================

print("\n" + "=" * 60)
print("6. ANOMALY DETECTION")
print("=" * 60)

anom_cols = [
    'WEATHER',
    'ROAD CONDITION',
    'VEHICLE TYPE',
    'CAUSE',
    'LOCATION CLEAN',
    'HOUR',
    'MONTH'
]

ac = view[anom_cols].copy()

ac_cat = [
    c for c in anom_cols
    if ac[c].dtype == 'object'
]

ac_num = [
    c for c in anom_cols
    if c not in ac_cat
]

for c in ac_cat:
    ac[c] = ac[c].fillna('Unknown').astype(str)

aprep = ColumnTransformer([
    (
        'cat',
        OneHotEncoder(handle_unknown='ignore'),
        ac_cat
    ),
    (
        'num',
        Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('scale', StandardScaler())
        ]),
        ac_num
    )
])

Xa = aprep.fit_transform(ac)

if len(view) >= 20:

    iso = IsolationForest(
        contamination=0.03,
        random_state=RANDOM_STATE
    )

    view['ANOMALY FLAG'] = np.where(
        iso.fit_predict(Xa) == -1,
        'Review unusual record',
        'Typical pattern'
    )

    display(
        view[
            view['ANOMALY FLAG'] == 'Review unusual record'
        ][
            [
                'DATE',
                'TIME',
                'LOCATION CLEAN',
                'WEATHER',
                'ROAD CONDITION',
                'VEHICLE TYPE',
                'CAUSE',
                'SEVERITY'
            ]
        ].head(30)
    )

else:
    view['ANOMALY FLAG'] = 'Not evaluated'
    print("Not enough records for anomaly detection.")

# ============================================================
# 7. DECISION TREE AND RANDOM FOREST CLASSIFICATION
# ============================================================

print("\n" + "=" * 60)
print("7. CLASSIFICATION MODELS")
print("=" * 60)

md = view.copy()

md['SEVERITY MODEL'] = md['SEVERITY'].fillna('Unknown')

counts = md['SEVERITY MODEL'].value_counts()

md.loc[
    md['SEVERITY MODEL'].isin(counts[counts < 2].index),
    'SEVERITY MODEL'
] = 'Rare/other'

features = [
    'WEATHER',
    'ROAD CONDITION',
    'VEHICLE TYPE',
    'CAUSE',
    'LOCATION CLEAN',
    'HOUR',
    'MONTH',
    'DAY OF WEEK',
    'WEEKEND'
]

X = md[features].copy()
y = md['SEVERITY MODEL'].copy()

catf = [
    'WEATHER',
    'ROAD CONDITION',
    'VEHICLE TYPE',
    'CAUSE',
    'LOCATION CLEAN',
    'DAY OF WEEK'
]

numf = ['HOUR', 'MONTH', 'WEEKEND']

for c in catf:
    X[c] = X[c].fillna('Unknown').astype(str)

for c in numf:
    X[c] = pd.to_numeric(X[c], errors='coerce')

if len(y) < 4 or y.nunique() < 2:
    raise ValueError(
        "Not enough records or target classes to train "
        "a meaningful classification model."
    )

Xtr, Xte, ytr, yte = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=RANDOM_STATE,
    stratify=y if y.value_counts().min() >= 2 else None
)

print("Training records:", len(Xtr))
print("Testing records:", len(Xte))

prep = ColumnTransformer([
    (
        'cat',
        Pipeline([
            ('imp', SimpleImputer(strategy='most_frequent')),
            ('oh', OneHotEncoder(handle_unknown='ignore'))
        ]),
        catf
    ),
    (
        'num',
        Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('scale', StandardScaler())
        ]),
        numf
    )
])

estimators = {
    'Decision Tree': DecisionTreeClassifier(
        random_state=RANDOM_STATE,
        class_weight='balanced',
        max_depth=6,
        min_samples_leaf=5,
        min_samples_split=10
    ),

    'Random Forest': RandomForestClassifier(
        n_estimators=250,
        random_state=RANDOM_STATE,
        class_weight='balanced_subsample',
        min_samples_leaf=2,
        n_jobs=-1
    )
}

fitted = {}
metric_rows = []

for name, est in estimators.items():

    pipe = Pipeline([
        ('prep', prep),
        ('model', est)
    ])

    pipe.fit(Xtr, ytr)

    predictions = pipe.predict(Xte)

    fitted[name] = pipe

    metric_rows.append({
        'Model': name,
        'Accuracy': accuracy_score(yte, predictions),
        'Precision macro': precision_score(
            yte, predictions,
            average='macro',
            zero_division=0
        ),
        'Recall macro': recall_score(
            yte, predictions,
            average='macro',
            zero_division=0
        ),
        'F1 macro': f1_score(
            yte, predictions,
            average='macro',
            zero_division=0
        ),
        'F1 weighted': f1_score(
            yte, predictions,
            average='weighted',
            zero_division=0
        )
    })

    print("\nMODEL:", name)
    print(classification_report(
        yte,
        predictions,
        zero_division=0
    ))

    ConfusionMatrixDisplay.from_predictions(
        yte,
        predictions,
        xticks_rotation=45
    )

    plt.title(name + ' Confusion Matrix')
    plt.tight_layout()
    plt.show()

metrics_df = pd.DataFrame(metric_rows)

print("\nMODEL COMPARISON")
display(metrics_df)

metrics_df.to_csv(
    'Bunawan_Model_Comparison.csv',
    index=False
)

best_name = metrics_df.sort_values(
    'F1 macro',
    ascending=False
).iloc[0]['Model']

best_model = fitted[best_name]

print("Model with highest test macro F1:", best_name)

# ============================================================
# 8. CROSS-VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("8. CROSS-VALIDATION")
print("=" * 60)

folds = min(5, int(ytr.value_counts().min()))

if folds >= 2:

    cv = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    cvrows = []

    for name, est in estimators.items():

        sc = cross_validate(
            Pipeline([
                ('prep', prep),
                ('model', est)
            ]),
            Xtr,
            ytr,
            cv=cv,
            scoring=[
                'accuracy',
                'precision_macro',
                'recall_macro',
                'f1_macro'
            ]
        )

        row = {'Model': name}

        for k, v in sc.items():

            if k.startswith('test_'):

                metric = k.replace('test_', '')

                row[metric + '_mean'] = float(np.mean(v))
                row[metric + '_std'] = float(np.std(v))

        cvrows.append(row)

    cv_df = pd.DataFrame(cvrows)

    display(cv_df.round(4))

    cv_df.to_csv(
        'Bunawan_Cross_Validation.csv',
        index=False
    )

else:
    cv_df = pd.DataFrame()

    print(
        "Cross-validation skipped because a training class "
        "contains fewer than two records."
    )

# ============================================================
# 9. ROAD INSPECTION PRIORITY ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("9. ROAD INSPECTION PRIORITY ANALYSIS")
print("=" * 60)


def severe_count(s):
    return int(
        s.fillna('')
        .astype(str)
        .str.lower()
        .str.contains(
            'fatal|major injury|serious injury|severe injury',
            regex=True
        ).sum()
    )


priority = view.groupby(
    'LOCATION CLEAN',
    dropna=False
).agg(
    records=('LOCATION CLEAN', 'size'),

    fatal_records=(
        'SEVERITY',
        lambda s: int(
            s.fillna('')
            .astype(str)
            .str.lower()
            .eq('fatal')
            .sum()
        )
    ),

    severe_records=('SEVERITY', severe_count),

    top_cause=(
        'CAUSE',
        lambda s: s.mode().iloc[0]
        if not s.mode().empty else 'Unknown'
    ),

    top_vehicle=(
        'VEHICLE TYPE',
        lambda s: s.mode().iloc[0]
        if not s.mode().empty else 'Unknown'
    ),

    top_weather=(
        'WEATHER',
        lambda s: s.mode().iloc[0]
        if not s.mode().empty else 'Unknown'
    ),

    top_road_condition=(
        'ROAD CONDITION',
        lambda s: s.mode().iloc[0]
        if not s.mode().empty else 'Unknown'
    ),

    night_records=(
        'HOUR',
        lambda s: int(((s >= 18) | (s < 6)).sum())
    )
).reset_index()

priority['count_norm'] = (
    priority['records'] /
    max(priority['records'].max(), 1)
)

priority['severe_norm'] = (
    priority['severe_records'] /
    max(priority['severe_records'].max(), 1)
)

priority['fatal_norm'] = (
    priority['fatal_records'] /
    max(priority['fatal_records'].max(), 1)
)

priority['screening_score'] = 100 * (
    0.50 * priority['count_norm']
    + 0.30 * priority['severe_norm']
    + 0.20 * priority['fatal_norm']
)

priority['inspection_tier'] = pd.cut(
    priority['screening_score'],
    [-0.01, 25, 50, 75, 100.01],
    labels=[
        'Routine review',
        'Monitor',
        'High-priority inspection',
        'Urgent engineering review'
    ]
)

priority = priority.sort_values(
    ['screening_score', 'records'],
    ascending=False
)

display(priority.head(30))

priority.to_csv(
    'Bunawan_RoadSign_Inspection_Priority.csv',
    index=False
)

plt.figure(figsize=(11, 6))

priority.head(15).sort_values(
    'screening_score'
).plot(
    kind='barh',
    x='LOCATION CLEAN',
    y='screening_score',
    legend=False,
    figsize=(11, 6)
)

plt.title('Road Inspection Priority Screening Score')
plt.xlabel('Score 0–100 (not a risk probability)')
plt.tight_layout()
plt.show()

print(
    "Reminder: The screening score supports prioritization only. "
    "Field inspections and engineering assessment are still required."
)

# ============================================================
# 10. ACCIDENT HOTSPOT MAPPING
# ============================================================

print("\n" + "=" * 60)
print("10. ACCIDENT HOTSPOT MAPPING")
print("=" * 60)

print(
    "Upload a location lookup CSV containing "
    "LOCATION CLEAN, LATITUDE, and LONGITUDE."
)

print(
    "Coordinates should be verified. "
    "Do not assign the same approximate coordinates to different "
    "crash sites and present them as exact crash locations."
)

geo_upload = files.upload()

geo_name = next(
    (n for n in geo_upload if n.lower().endswith('.csv')),
    None
)

mapped = pd.DataFrame()

if geo_name:

    geo = pd.read_csv(geo_name)

    geo.columns = [
        str(c).strip().upper()
        for c in geo.columns
    ]

    needed = {
        'LOCATION CLEAN',
        'LATITUDE',
        'LONGITUDE'
    }

    if not needed.issubset(geo.columns):
        raise ValueError(
            f"Coordinate lookup must contain {needed}"
        )

    geo['LOCATION CLEAN'] = (
        geo['LOCATION CLEAN'].map(normalize_location)
    )

    geo['LATITUDE'] = pd.to_numeric(
        geo['LATITUDE'],
        errors='coerce'
    )

    geo['LONGITUDE'] = pd.to_numeric(
        geo['LONGITUDE'],
        errors='coerce'
    )

    geo = (
        geo.dropna(subset=['LATITUDE', 'LONGITUDE'])
        .drop_duplicates('LOCATION CLEAN')
    )

    mapped = priority.merge(
        geo,
        on='LOCATION CLEAN',
        how='inner'
    )

    print(
        "Matched locations:",
        len(mapped),
        "of",
        len(priority)
    )

    if len(mapped):

        m = folium.Map(
            location=[
                mapped['LATITUDE'].mean(),
                mapped['LONGITUDE'].mean()
            ],
            zoom_start=12
        )

        HeatMap(
            mapped[
                ['LATITUDE', 'LONGITUDE', 'records']
            ].values.tolist(),
            radius=18,
            blur=15
        ).add_to(m)

        for _, r in mapped.iterrows():

            rc = (
                r['ROAD CLASS']
                if 'ROAD CLASS' in mapped.columns
                else 'Unverified'
            )

            folium.CircleMarker(
                [
                    r['LATITUDE'],
                    r['LONGITUDE']
                ],
                radius=max(
                    4,
                    min(14, 3 + math.sqrt(r['records']))
                ),
                tooltip=str(r['LOCATION CLEAN']),
                popup=(
                    f"Records: {r['records']}<br>"
                    f"Severe: {r['severe_records']}<br>"
                    f"Score: {r['screening_score']:.1f}<br>"
                    f"Road class: {rc}"
                ),
                fill=True
            ).add_to(m)

        display(m)

        m.save('Bunawan_RoadAccident_Hotspots.html')

        print("Map saved: Bunawan_RoadAccident_Hotspots.html")

    else:
        print(
            "No matching locations. "
            "Check spelling and coordinate lookup labels."
        )

else:
    print("No coordinate lookup uploaded. Map was not generated.")

# ============================================================
# 11. CREATE STREAMLIT DASHBOARD FILE
# ============================================================

print("\n" + "=" * 60)
print("11. CREATE STREAMLIT DASHBOARD")
print("=" * 60)

dashboard_code = r'''
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="Bunawan Road Safety Dashboard",
    layout="wide"
)

st.title("Bunawan Road Accident & Road-Sign Decision Support")

st.caption(
    "Historical screening tool only. "
    "Verify locations and engineering requirements before "
    "installing road signs."
)

up = st.file_uploader(
    "Upload cleaned accident CSV",
    type=["csv"]
)

if up:

    d = pd.read_csv(up)

    d.columns = [
        str(c).strip().upper().replace("_", " ")
        for c in d.columns
    ]

    date_col = (
        "DATE PARSED"
        if "DATE PARSED" in d.columns
        else "DATE"
    )

    if date_col in d.columns:

        d[date_col] = pd.to_datetime(
            d[date_col],
            errors="coerce",
            dayfirst=True
        )

        d["YEAR FILTER"] = d[date_col].dt.year

    else:
        d["YEAR FILTER"] = np.nan

    loc_col = (
        "LOCATION CLEAN"
        if "LOCATION CLEAN" in d.columns
        else "LOCATION"
    )

    if loc_col not in d.columns:
        st.error("The dataset does not contain a location column.")
        st.stop()

    years = sorted(
        d["YEAR FILTER"].dropna().astype(int).unique().tolist()
    )

    locs = sorted(
        d[loc_col].dropna().astype(str).unique().tolist()
    )

    with st.sidebar:

        st.header("Filters")

        ys = st.multiselect(
            "Year",
            years,
            default=years
        )

        ls = st.multiselect(
            "Location",
            locs,
            default=locs
        )

    f = d.copy()

    if ys:
        f = f[f["YEAR FILTER"].isin(ys)]
    else:
        f = f.iloc[0:0]

    if ls:
        f = f[f[loc_col].astype(str).isin(ls)]
    else:
        f = f.iloc[0:0]

    sev = (
        f["SEVERITY"].fillna("").astype(str).str.lower()
        if "SEVERITY" in f.columns
        else pd.Series("", index=f.index)
    )

    a, b, c, e = st.columns(4)

    a.metric("Records", f"{len(f):,}")
    b.metric("Locations", f"{f[loc_col].nunique():,}")
    c.metric("Fatal records", int(sev.eq("fatal").sum()))
    e.metric(
        "Major injury records",
        int(sev.str.contains("major injury").sum())
    )

    left, right = st.columns(2)

    if "SEVERITY" in f.columns:

        s = (
            f["SEVERITY"]
            .fillna("Unknown")
            .value_counts()
            .rename_axis("Severity")
            .reset_index(name="Records")
        )

        left.plotly_chart(
            px.bar(
                s,
                x="Severity",
                y="Records",
                title="Severity Categories"
            ),
            use_container_width=True
        )

    if "CAUSE" in f.columns:

        q = (
            f["CAUSE"]
            .fillna("Unknown")
            .value_counts()
            .head(12)
            .rename_axis("Cause")
            .reset_index(name="Records")
        )

        right.plotly_chart(
            px.bar(
                q,
                x="Records",
                y="Cause",
                orientation="h",
                title="Top Entered Causes"
            ),
            use_container_width=True
        )

    h = (
        f[loc_col]
        .fillna("Unknown")
        .value_counts()
        .head(20)
        .rename_axis("Location")
        .reset_index(name="Records")
    )

    st.plotly_chart(
        px.bar(
            h,
            x="Records",
            y="Location",
            orientation="h",
            title="Top Recorded Locations"
        ),
        use_container_width=True
    )

    st.subheader("Inspection Priority Screening")

    if "SEVERITY" in f.columns:

        x = f.copy()

        x["_fatal"] = (
            x["SEVERITY"]
            .fillna("")
            .astype(str)
            .str.lower()
            .eq("fatal")
            .astype(int)
        )

        x["_severe"] = (
            x["SEVERITY"]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.contains(
                "fatal|major injury|serious injury|severe injury",
                regex=True
            )
            .astype(int)
        )

        p = x.groupby(loc_col).agg(
            records=(loc_col, "size"),
            fatal_records=("_fatal", "sum"),
            severe_records=("_severe", "sum")
        ).reset_index()

        if len(p):

            p["screening_score"] = 100 * (
                0.5 * p["records"] / max(p["records"].max(), 1)
                + 0.3 * p["severe_records"] / max(
                    p["severe_records"].max(), 1
                )
                + 0.2 * p["fatal_records"] / max(
                    p["fatal_records"].max(), 1
                )
            )

            p = p.sort_values(
                "screening_score",
                ascending=False
            )

            st.dataframe(p, use_container_width=True)

            st.download_button(
                "Download Inspection Priority CSV",
                p.to_csv(index=False).encode(),
                "road_sign_inspection_priority.csv",
                "text/csv"
            )

    st.warning(
        "The screening score is not a probability or an automatic "
        "sign-installation warrant. Validate exact crash sites, road "
        "authority, sight distance, speeds, existing signs and "
        "applicable engineering standards."
    )

else:

    st.info("Upload RACC_cleaned_audit.csv to start.")
'''

with open(
    'bunawan_road_safety_dashboard.py',
    'w',
    encoding='utf-8'
) as f:
    f.write(dashboard_code)

print("Dashboard file created: bunawan_road_safety_dashboard.py")

# ============================================================
# 12. DOWNLOAD OUTPUT FILES
# ============================================================

print("\n" + "=" * 60)
print("12. DOWNLOAD OUTPUT FILES")
print("=" * 60)

output_files = [
    'RACC_cleaned_audit.csv',
    'Bunawan_RoadSign_Inspection_Priority.csv',
    'Bunawan_Model_Comparison.csv',
    'bunawan_road_safety_dashboard.py'
]

if not cv_df.empty:
    output_files.append('Bunawan_Cross_Validation.csv')

if os.path.exists('Bunawan_RoadAccident_Hotspots.html'):
    output_files.append('Bunawan_RoadAccident_Hotspots.html')

for filename in output_files:

    if os.path.exists(filename):

        print("Downloading:", filename)
        files.download(filename)

print("\nALL PROCESSING STEPS COMPLETED!")
