# %% read csv file
import pandas as pd


df = pd.read_csv("2012.csv")

# %% Profile the data set
# import pandas_profiling as pp

# pp.ProfileReport(pd.read_csv("2012.csv"))

# %% describe the dataset
df.describe()


# %% Check for null values
import numpy as np
df.isnull().sum

# %% Check for missing values
df.info()

# %% handling missing values
df =df.dropna()

# %%
cols = [
    "datestop",
    "timestop",
]

for col in cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# %%
df["datestop"] =df["datestop"].astype(str).str.zfill(8)
df["timestop"] =df["timestop"].astype(str).str.zfill(4)

from datetime import datetime

def make_datetime(datestop, timestop):
    year = int(datestop[-4:])
    month = int(datestop[:2])
    day = int(datestop[2:4])

    hour = int(timestop[:2])
    minute = int(timestop[2:])


    return datetime(year, month, day, hour, minute)


# %%
df["datetime"] =df.apply(lambda r: make_datetime(r["datestop"], r["timestop"]), axis=1)






# %% make sure numeric columns have numbers, remove rows that are not
cols = [
    "perobs",
    "perstop",
    "age",
    "weight",
    "ht_feet",
    "ht_inch",
    "datestop",
    "timestop",
    "xcoord",
    "ycoord",
]

for col in cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna()



# %% convert all value to label in the dataframe, remove rows that cannot be mapped
import numpy as np
from tqdm import tqdm

value_label = pd.read_excel(
    "2012 SQF File Spec.xlsx",
    sheet_name="Value Labels",
    skiprows=range(4)
)
value_label["Field Name"] = value_label["Field Name"].fillna(
    method="ffill"
)
value_label["Field Name"] = value_label["Field Name"].str.lower()
value_label["Value"] = value_label["Value"].fillna(" ")
value_label = value_label.groupby("Field Name").apply(
    lambda x: dict([(row["Value"], row["Label"]) for row in x.to_dict("records")])
)

cols = [col for col in df.columns if col in value_label]

for col in tqdm(cols):
    df[col] = df[col].apply(
        lambda val: value_label[col].get(val, np.nan)
    )

df["trhsloc"] = df["trhsloc"].fillna("P (unknown)")
df = df.dropna()


# %% convert xcoord and ycoord to (lon, lat)
import pyproj

srs = "+proj=lcc +lat_1=41.03333333333333 +lat_2=40.66666666666666 +lat_0=40.16666666666666 +lon_0=-74 +x_0=300000.0000000001 +y_0=0 +ellps=GRS80 +datum=NAD83 +to_meter=0.3048006096012192 +no_defs"
p = pyproj.Proj(srs)

df["coord"] = df.apply(
    lambda r: p(r["xcoord"], r["ycoord"], inverse=True), axis=1
)

# %% convert height in feet/inch to cm
df["height"] = (df["ht_feet"] * 12 + df["ht_inch"]) * 2.54


# %% remove outlier
df = df[(df["age"] <= 90) & (df["age"] >= 12)]
df = df[(df["weight"] <= 350) & (df["weight"] >= 50)]


# %% delete columns that are not needed
df = df.drop(
    columns=[
        # processed columns
        "datestop",
        "timestop",
        "ht_feet",
        "ht_inch",
        "xcoord",
        "ycoord",

        # not useful
        "year",
        "recstat",
        "crimsusp",
        "dob",
        "ser_num",
        "arstoffn",
        "sumoffen",
        "compyear",
        "comppct",
        "othfeatr",
        "adtlrept",
        "dettypcm",
        "linecm",
        "repcmd",
        "revcmd",

        # location of stop
        # only use coord and city
        "addrtyp",
        "rescode",
        "premtype",
        "premname",
        "addrnum",
        "stname",
        "stinter",
        "crossst",
        "aptnum",
        "state",
        "zip",
        "addrpct",
        "sector",
        "beat",
        "post",
    ]
)


# %%
import folium
import seaborn as sns
import matplotlib.pyplot as plt


ax = sns.countplot(df["datetime"].dt.month)
plt.axis('on')

ax.set(xlabel='Month')
ax.set(ylabel='Stops made')
ax.set(title='Frequency of Stops made by Month')


#%%
ax = sns.scatterplot(x="age", y="eyecolor", hue="sex", data=df)
plt.xlabel("Age")
plt.ylabel("Weight in lbs")
ax.plot(legend=None)
plt.show()

 # %%
ax = sns.countplot(df["datetime"].dt.weekday)
# The day of the week with Monday=0, Sunday=6. See https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DatetimeIndex.weekday.html
ax.set_xticklabels(
    ["Mon", "Tue", "Wed", "Thur", "Fri", "Sat", "Sun"]
)
ax.set(xlabel='Day of Week')
ax.set(ylabel='Stops Made')
ax.set(title="Frequency of stops by Day")

#%%
ax = sns.distplot(df["datetime"].dt.hour)
ax.set(xlabel='Hour of Day')
ax.set(ylabel='Stops Made')
ax.set(title="Frequency of stops by Hour")

# %%
ax = sns.countplot(df["datetime"].dt.hour)
ax.set(xlabel='Hour of Day')
ax.set(ylabel='Stops Made')
ax.set(title="Frequency of stops by Hour")

# %%
ax = sns.countplot(data=df,x="race")
ax.set_xticklabels(
    ax.get_xticklabels(), rotation=45, fontsize=5
)
for p in ax.patches:
    percentage = p.get_height() * 100 / df.shape[0]
    txt = f"{percentage:.1f}%"
    x_loc = p.get_x()
    y_loc = p.get_y() + p.get_height()
    ax.text(x_loc, y_loc, txt)

ax.set(xlabel='Race of Individual')
ax.set(ylabel='Stops Made')
ax.set(title="Number of stops by race")


#%%
ax = sns.countplot(data=df,x="race", hue="sex")
ax.set_xticklabels(
    ax.get_xticklabels(), rotation=45, fontsize=5
)
#for p in ax.patches:
#    percentage = p.get_height() * 100 / df.shape[0]
#    txt = f"{percentage:.1f}%"
#    x_loc = p.get_x()
#    y_loc = p.get_y() + p.get_height()
#    ax.text(x_loc, y_loc, txt)

ax.set(xlabel='Race and Sex')
ax.set(ylabel='Stops Made')
ax.set(title="Number of stops by race and sex")
# %%
ax = sns.countplot(df["race"], hue=df["sex"])
ax.set_xticklabels(
    ax.get_xticklabels(), rotation=45, fontsize=5
)
plt.legend 
#%%
ax = sns.countplot(df["sex"], hue=df["build"])
ax.set_xticklabels(
    ax.get_xticklabels(), rotation=45, fontsize=5
)

#%%
ax = sns.countplot(df["race"], hue=df["build"])
ax.set_xticklabels(
    ax.get_xticklabels(), rotation=45, fontsize=5
)
#%%
ax = sns.countplot(df["city"], hue=df["race"])
ax.set_xticklabels(
    ax.get_xticklabels(), rotation=45, fontsize=5
)

#%%
ax = sns.countplot(df["city"], hue=df["sex"])
ax.set_xticklabels(
    ax.get_xticklabels(), rotation=45, fontsize=5
)
#%% 
ax = sns.distplot(df["age"], bins=20)

plt.ylabel("Fraction of total Stops")
plt.xlabel("Age")
plt.title("SQF stops by Age")

#%%
ax = sns.distplot(df["weight"], bins=20)

plt.ylabel("Fraction of Stops")
plt.xlabel("Weight")
plt.title("SQF stops by Weight")
# %%
ax = sns.countplot(df["race"], hue=df["city"])
ax.set_xticklabels(
    ax.get_xticklabels(), rotation=45, fontsize=5
)
plt.ylabel("Stops made")
plt.xlabel("Race")
plt.title("SQF stops by Race and City")

# %%
nyc = (40.730610, -73.935242)

m = folium.Map(location=nyc)


# %%
for coord in df.loc[df["detailcm"]=="LOITERING", "coord"]:
    folium.CircleMarker(
        location=(coord[1], coord[0]), radius=1, color="orange"
    ).add_to(m)

m


# %%
for coord in df.loc[df["detailcm"]=="MAKING GRAFFITI", "coord"]:
    folium.CircleMarker(
        location=(coord[1], coord[0]), radius=1, color="yellow"
    ).add_to(m)

m
# %%
for coord in df.loc[df["detailcm"]=="RIOT", "coord"]:
    folium.CircleMarker(
        location=(coord[1], coord[0]), radius=1, color="BLUE"
    ).add_to(m)

m

# %% Correlation between attributes

def show_correlations(dataframe, show_chart = True):
    fig = plt.figure(figsize = (12,12))
    corr = dataframe.corr()
    if show_chart == True:
        sns.heatmap(corr,
                    xticklabels=corr.columns.values,
                    yticklabels=corr.columns.values,
                    annot=True)
    return corr

correlation_df = show_correlations(df,show_chart=True)

#%% Scatterplot????

# %%

forces = [col for col in df.columns if col.startswith("pf_")]
stop = [col for col in df.columns if col.startswith("cs_")]
frisk = [col for col in df.columns if col.startswith("rf_")]

result = ["arstmade", "sumissue", "searched", "frisked"]

subset = df[forces + stop + frisk]
subset = (subset == "YES").astype(int)

f,ax = plt.subplots(figsize=(25, 25))
sns.heatmap(subset.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

#sns.heatmap(subset.corr())
# %%

forces = [col for col in df.columns if col.startswith("pf_")]
stop = [col for col in df.columns if col.startswith("cs_")]
frisk = [col for col in df.columns if col.startswith("rf_")]

result = ["arstmade", "sumissue", "searched", "frisked"]

subset = df[forces + stop + frisk]
subset = (subset == "YES").astype(int)

f,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(subset.corr(), linewidths=.2, fmt= '.1f',ax=ax)

# %%
df.to_pickle("sqf.pkl")



