# %% read dataframe from part1
import pandas as pd

df = pd.read_pickle("sqf.pkl")


# %% pick a criminal code you want, probably not the one with high counts, in case the folium map hangs
df_cah = df[df["detailcm"] == "CREATING A HAZARD"]

df_cah["lat"] = df["coord"].apply(lambda val: val[1])
df_cah["lon"] = df["coord"].apply(lambda val: val[0])


# %% run hierarchical clustering on a range of arbitrary values
# record the silhouette_score and find the best number of clusters
# THIS STEP IS FOR DETERMINING THE NUMBER OF CLUSTERS TO USE (SIL SCORE)
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import matplotlib.pyplot as plt

silhouette_scores, labels = {}, {}
num_city = df["city"].nunique()
num_pct = df["pct"].nunique()
step = 10
 
for k in tqdm(range(num_city, num_pct, step)):
    c = AgglomerativeClustering(n_clusters=k)
    y = c.fit_predict(df_cah[["lat", "lon"]])
    silhouette_scores[k] = silhouette_score(df_cah[["lat", "lon"]], y)
    labels[k] = y



# %% plot the silhouette_scores agains different numbers of clusters
import seaborn as sns

ax = sns.lineplot(x=list(silhouette_scores.keys()), y=list(silhouette_scores.values()),)
ax.get_figure().savefig("trend.png", bbox_inches="tight", dpi=400)
plt.title("Silhouette Method for Optimal k")
plt.ylabel("Silhouette Score")
plt.xlabel("k")

# %% visualize the clustering label on a map
# using the color palette from seaborn
import folium

nyc = (40.730610, -73.935242)
m = folium.Map(location=nyc)

best_k = max(silhouette_scores, key=lambda k: silhouette_scores[k])
df_cah["label"] = labels[best_k]


colors = sns.color_palette("bright", best_k).as_hex()

for row in tqdm(df_cah[["lat", "lon", "label"]].to_dict("records")):
    folium.CircleMarker(
        location=(row["lat"], row["lon"]), radius=1, color=colors[row["label"]]
    ).add_to(m)

m


# %% find reason for stop columns
css = [col for col in df.columns if col.startswith("cs_")]


# %% run dbscan on reason for stop columns - if DBSCAN doesnt work well, try KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from tqdm import tqdm

c = DBSCAN()
x = df_cah[css] == "YES"
y = c.fit_predict(x)
print(silhouette_score(x, y))


#%% Visualizing the DBScan
import numpy as np

nyc = (40.730610, -73.935242)
m = folium.Map(location=nyc)
df_cah["label"] = y
colors = sns.color_palette("bright", len(np.unique(y))).as_hex()
for row in tqdm(df_cah[["lat", "lon", "label"]].to_dict("records")):
    folium.CircleMarker(
        location=(row["lat"], row["lon"]),
        radius=0.1,
        color=colors[row["label"]],
        alpha=0.6,
    ).add_to(m)
m


#%%
nyc = (40.730610, -73.935242)
m = folium.Map(location=nyc)

best_k = max(silhouette_scores, key=lambda k: silhouette_scores[k])
df_cah["label"] = labels[best_k]


colors = sns.color_palette("bright", best_k).as_hex()

for row in tqdm(df_cah[["lat", "lon", "label"]].to_dict("records")):
    folium.CircleMarker(
        location=(row["lat"], row["lon"]), radius=1, color=colors[row["label"]]
    ).add_to(m)

m
# %% find reason for stop columns
css = [col for col in df.columns if col.startswith("cs_")]
x = df_cah[css] == "YES"
  
# %% run KMeans clustering on a range of arbitrary values
# record the silhouette_score and find the best number of clusters
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm
  
silhouette_scores, labels = {}, {}
num_city = df["city"].nunique()
num_pct = df["pct"].nunique()
step = 10
x["lat"] = df_cah["lat"]
x["lon"] = df_cah["lon"]
  
for k in tqdm(range(num_city, num_pct, step)):
    km = KMeans(n_clusters=k)
    y = km.fit_predict(x)
    silhouette_scores[k] = silhouette_score(x, y)
    labels[k] = y


#%% Visualizing Kmeans

nyc = (40.730610, -73.935242)
m = folium.Map(location=nyc)
df_cah["label"] = y
colors = sns.color_palette("bright", len(np.unique(y))).as_hex()
for row in tqdm(df_cah[["lat", "lon", "label"]].to_dict("records")):
    folium.CircleMarker(
        location=(row["lat"], row["lon"]),
        radius=0.1,
        color=colors[row["label"]],
        alpha=0.6,
    ).add_to(m)
m


# %% plot the silhouette_scores agains different numbers of clusters
import seaborn as sns
  
ax = sns.lineplot(x=list(silhouette_scores.keys()), y=list(silhouette_scores.values()),)

plt.title("Silhouette Method for Optimal k")
plt.ylabel("Silhouette Score")
plt.xlabel("k")
# %% visualize the clustering label on a map
# using the color palette from seaborn
import folium
  
nyc = (40.730610, -73.935242)
m = folium.Map(location=nyc)
  
best_k = max(silhouette_scores, key=lambda k: silhouette_scores[k])
x["label"] = labels[best_k]
  
colors = sns.color_palette("bright", best_k).as_hex()
  
for row in tqdm(x[["lat", "lon", "label"]].to_dict("records")):
    folium.CircleMarker(
        location=(row["lat"], row["lon"]), radius=1, color=colors[row["label"]]
    ).add_to(m)
  
m



# %% visualize the new clustering label on a map
import numpy as np

nyc = (40.730610, -73.935242)
m = folium.Map(location=nyc)
df_cah["label"] = y
colors = sns.color_palette("bright", len(np.unique(y))).as_hex()
for row in tqdm(df_cah[["lat", "lon", "label"]].to_dict("records")):
    folium.CircleMarker(
        location=(row["lat"], row["lon"]),
        radius=0.1,
        color=colors[row["label"]],
        alpha=0.3,
    ).add_to(m)
m


# %% pick some of the labels to see if there's any location wise insight
import numpy as np

nyc = (40.730610, -73.935242)
m = folium.Map(location=nyc)
df_cah["label"] = y
colors = sns.color_palette("bright", len(np.unique(y))).as_hex()
for row in tqdm(
    df_cah.loc[df_cah["label"] == y[55], ["lat", "lon", "label"]].to_dict(
        "records"
    )
):
    folium.CircleMarker(
        location=(row["lat"], row["lon"]),
        radius=0.1,
        color=colors[row["label"]],
        alpha=0.3,
    ).add_to(m)
m


# %% pick some of the labels to see if there's any location wise insight
import numpy as np

nyc = (40.730610, -73.935242)
m = folium.Map(location=nyc)
df_cah["label"] = y
colors = sns.color_palette("bright", len(np.unique(y))).as_hex()
for row in tqdm(
    df_cah.loc[df_cah["label"] == y[7], ["lat", "lon", "label"]].to_dict(
        "records"
    )
):
    folium.CircleMarker(
        location=(row["lat"], row["lon"]),
        radius=0.1,
        color=colors[row["label"]],
        alpha=0.3,
    ).add_to(m)
m


# %% pick some of the labels to see if there's any location wise insight
import numpy as np

nyc = (40.730610, -73.935242)
m = folium.Map(location=nyc)
df_assault["label"] = y
colors = sns.color_palette("bright", len(np.unique(y))).as_hex()
for row in tqdm(
    df_assault.loc[df_assault["label"] == y[6], ["lat", "lon", "label"]].to_dict(
        "records"
    )
):
    folium.CircleMarker(
        location=(row["lat"], row["lon"]),
        radius=0.1,
        color=colors[row["label"]],
        alpha=0.3,
    ).add_to(m)
m


# %%
