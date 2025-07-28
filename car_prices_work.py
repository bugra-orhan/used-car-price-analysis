# 0) Import required libraries 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1) Read and describe the data
df = pd.read_csv("car_prices.csv")
print(df.head())
print(df.info())
print(df.describe())

# 2-a) Manipulate the data
df["make_model"] = df["make"].astype(str)+" "+df["model"]

cols = list(df.columns)
last_col=cols[-1]
new_order = [cols[0],last_col] + cols[1:-1]
df = df[new_order]

# 558837 entries, so columns do not have N/A: year, state, seller
# this dataset includes the cars produced between 1982 and 2015.
# thx to describe() method, we can distinct numerical (even ordinal) and categorical attributes. only we have to make df["year"] as str 
df['saledate_clean'] = df['saledate'].str.extract(r'(\w{3} \w{3} \d{2} \d{4} \d{2}:\d{2}:\d{2})')[0]
df['saledate_clean'] = pd.to_datetime(df['saledate_clean'], format='%a %b %d %Y %H:%M:%S')

df['state'] = df['state'].apply(lambda x: x.upper().strip() if isinstance(x, str) else x)

df['seller'] = df['seller'].fillna('').astype(str)
df['seller'] = df['seller'].str.lower().str.replace(r'[^a-z0-9]', ' ', regex=True)
df['seller'] = df['seller'].str.split().str.join(' ')

df['year_sold'] = round(df['saledate_clean'].dt.year,0).astype('Int64').astype(str)
df['month_sold'] = round(df['saledate_clean'].dt.month,0).astype('Int64').astype(str)
df['period'] = df['year_sold'].astype(str) + '-' + df['month_sold'].astype(str).str.zfill(2)


df['body'] = df['body'].apply(lambda x: x.lower().strip().capitalize() if isinstance(x, str) else x)
df['transmission'] = df['transmission'].apply(lambda x: x.lower().strip().capitalize() if isinstance(x, str) else x)
df['seller'] = df['seller'].apply(lambda x: x.lower().strip().capitalize() if isinstance(x, str) else x)

# 2-b) Drop the irrelevant columns.
df.drop(["make","model","saledate","saledate_clean","year_sold","month_sold"],axis=1,inplace=True)

# 3) Prepare the data in order to visualize distribution of each column. 
# 3-a) Separate numerical and categorical columns.
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
numerical_cols = numerical_cols.drop(["year"])
categorical_cols = df.select_dtypes(include=['object', 'category', 'string']).columns
categorical_cols = categorical_cols.drop(["period"])

# 3-b) Create a plot for numerical columns.

fig, axes = plt.subplots(2, 2, figsize=(12, 8))  # 2x2 grid
axes = axes.flatten()  

for i, col in enumerate(numerical_cols):
    sns.boxplot(y=df[col], ax=axes[i])
    axes[i].set_title(f'Boxplot - {col}', fontsize=10)

plt.tight_layout()
plt.show()

# 3-c) Create a plot for categorical columns.
categorical_cols_clean = [col for col in categorical_cols if col!='vin']
fig, axes = plt.subplots(2, 4, figsize=(22, 8))
axes = axes.flatten()

for i, col in enumerate(categorical_cols_clean):
    sns.countplot(data=df, x=col, ax=axes[i], order=df[col].value_counts().index[:10])
    axes[i].set_title(f'Countplot - {col}',fontsize=10)
    axes[i].tick_params(axis='x', rotation=90)
    axes[i].set_xlabel("")
plt.tight_layout()
plt.show()

# 3-d) Create the barplot for monthly sales and lineplot for production year of cars. 
df['period'] = pd.to_datetime(df['period'], format='%Y-%m', errors='coerce')
df_period = df.dropna(subset=['period'])
df_period.groupby('period').size().sort_index().plot(
    kind='line',
    marker='o',
    figsize=(14,5),
    title='Monthly Sales (Jan 2014 - Jul 2015)'
)
plt.show()

df['year'].value_counts().sort_index().plot(
    kind='bar', figsize=(14,5), title='Production Year vs Sales'
)
plt.show()

# 3-e) Infer that a unique car can be sold for many times.
print("Sales of a specific car: ")
print(df[df["vin"]=="wbanv13588cz57827"][["mmr","sellingprice","period"]]) 

# 4) Visualize the effect of a column on one other column.

# 4-a) Production year vs. selling price

df.groupby('year')['sellingprice'].mean().sort_index().plot(
    kind='line', marker='o', figsize=(12,5), title='Average Selling Price by Year'
)

plt.tight_layout()
plt.show()

# 4-b) Brand & model vs. selling price

top_models = df['make_model'].value_counts().nlargest(15).index
df_top_models = df[df['make_model'].isin(top_models)]

avg_prices = df_top_models.groupby('make_model')['sellingprice'].mean().sort_values(ascending=False)

plt.figure(figsize=(14,6))
sns.barplot(x=avg_prices.index, y=avg_prices.values)
plt.title('Average Selling Price by Top 15 Vehicle Models')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# 4-c) Body vs. selling price

top_bodies = (
    df.groupby('body')['sellingprice']
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(12,5))
sns.barplot(x=top_bodies.index, y=top_bodies.values)
plt.title('Top 10 Body Types by Average Selling Price')
plt.xlabel('Body Type')
plt.ylabel('Average Selling Price')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 4-d) Odometer vs. selling price

plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x='odometer', y='sellingprice', alpha=0.3)
plt.title('Selling Price vs Odometer')
plt.xlabel('Odometer')
plt.ylabel('Selling Price')
plt.tight_layout()
plt.show()


# 4-e) Odometer vs condition

plt.figure(figsize=(10,6))
sns.violinplot(data=df, x='condition', y='odometer')
plt.title('Odometer by Vehicle Condition')
plt.xlabel('Condition')
plt.ylabel('Odometer')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# 4-f) Same make_models in different states

top_models = df['make_model'].value_counts().nlargest(6).index
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, model in enumerate(top_models):
    model_data = df[df['make_model'] == model]
    
    # Sadece en pahalı 10 eyalet
    avg_by_state = (
        model_data.groupby('state')['sellingprice']
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )

    sns.barplot(x=avg_by_state.index, y=avg_by_state.values, ax=axes[i])
    axes[i].set_title(f'{model} - Top 10 States by Avg Price',fontsize=10)
    axes[i].tick_params(axis='x', rotation=45,labelsize=8)
    axes[i].set_xlabel("")

plt.tight_layout()
plt.show()


