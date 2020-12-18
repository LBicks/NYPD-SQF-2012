# %% read dataframe from part1
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


df = pd.read_pickle("sqf.pkl")


# %% select some yes/no columns to convert into a dataframe of boolean values
pfs = [col for col in df.columns if col.startswith("pf_")]

armed = [
    "contrabn",
    "pistol",
    "riflshot",
    "asltweap",
    "knifcuti",
    "machgun",
    "othrweap",
]

x = df[pfs + armed]
x = x == "YES"


# %% create a new column to represent whether a person is armed
x["armed"] = (
    x["contrabn"]
    | x["pistol"]
    | x["riflshot"]
    | x["asltweap"]
    | x["knifcuti"]
    | x["machgun"]
    | x["othrweap"]
)

# %% select some categorical columns and do one hot encoding
# add some more columns in here, maybe AGE, WEIGHT, BODY TYPE, etc. (Matt did eye color, build)
for val in df["race"].unique():
    x[f"race_{val}"] = df["race"] == val

for val in df["age"].unique():
    x[f"age_{val}"] = df["age"] == val

for val in df["build"].unique():
    x[f"build_{val}"] = df["build"] == val

for val in df["city"].unique():
    x[f"city_{val}"] = df["city"] == val

for val in df["sex"].unique():
    x[f"sex_{val}"] = df["sex"] == val

for val in df["eyecolor"].unique():
    x[f"eyecolor_{val}"] = df["eyecolor"] == val

# %% apply frequent itemsets mining, make sure you play around of the support level

frequent_itemsets = apriori(x, min_support=0.015, use_colnames=True)
#frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

#%%
length = frequent_itemsets["itemsets"].apply(len)

print(f"{(length>2).sum()} itemsets that have 2 or more variables")

frequent_itemsets[length == length.max()]
# %% apply association rules mining 
rules = association_rules(frequent_itemsets, min_threshold=0.7)
rules
rules.sort_values(['lift'], ascending=False)

# %% sort rules by confidence and select rules within "armed" in it
rules.sort_values("confidence", ascending=False)

#%%
rules.sort_values("confidence", ascending=False)[
    rules.apply(
        lambda r: "armed" in r["antecedents"]
    or "armed" in r["consequents"],
    axis=1,
    )
]

# %%
import seaborn as sns
import matplotlib.pyplot as plt

ax = sns.scatterplot(
x="support", y="confidence", alpha=0.7, data=rules
)
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.title("Support vs Confidence")
plt.show()



# %%
ax = sns.scatterplot(
x="lift", y="support", alpha=0.7, data=rules
)
plt.xlabel("")
plt.ylabel("")
plt.title("Scatterplot of")
plt.show()

# %%
