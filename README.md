# Analysis of Online Recipe Ratings

**Name**: Christopher Brown

<details>
<summary>Click to View Initial Libraries for this Project</summary>
    
```python
import pandas as pd
import numpy as np
from pathlib import Path

import plotly.express as px
pd.options.plotting.backend = 'plotly'

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import plotly.graph_objects as go
from plotly.subplots import make_subplots
```
</details>

## Introduction

The data for this project consists of a set of recipes from food.com and a set of recipe ratings and reviews. I intend to answer the question: What is the relationship between cooking time and average ratings for recipes?

## Data Cleaning and Exploratory Data Analysis

<details>
<summary>Click to View Code</summary>

```python
# Load csv's as df's
ratings = pd.read_csv("RAW_interactions.csv")
recipes = pd.read_csv("RAW_recipes.csv")

# Calc avg_ratings and add it as a column to recipes
merged_df = recipes.merge(ratings, left_on='id', right_on='recipe_id', how='left')
merged_df['rating'] = merged_df['rating'].replace(0, np.nan)
avg_rating_per_recipe = merged_df.groupby('id')['rating'].mean()
recipes['avg_rating'] = recipes['id'].map(avg_rating_per_recipe)
```


```python
merged_df.head(2)
```
</details>



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>id</th>
      <th>minutes</th>
      <th>contributor_id</th>
      <th>submitted</th>
      <th>tags</th>
      <th>nutrition</th>
      <th>n_steps</th>
      <th>steps</th>
      <th>description</th>
      <th>ingredients</th>
      <th>n_ingredients</th>
      <th>user_id</th>
      <th>recipe_id</th>
      <th>date</th>
      <th>rating</th>
      <th>review</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1 brownies in the world    best ever</td>
      <td>333281</td>
      <td>40</td>
      <td>985201</td>
      <td>2008-10-27</td>
      <td>['60-minutes-or-less', 'time-to-make', 'course...</td>
      <td>[138.4, 10.0, 50.0, 3.0, 3.0, 19.0, 6.0]</td>
      <td>10</td>
      <td>['heat the oven to 350f and arrange the rack i...</td>
      <td>these are the most; chocolatey, moist, rich, d...</td>
      <td>['bittersweet chocolate', 'unsalted butter', '...</td>
      <td>9</td>
      <td>386585.0</td>
      <td>333281.0</td>
      <td>2008-11-19</td>
      <td>4.0</td>
      <td>These were pretty good, but took forever to ba...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1 in canada chocolate chip cookies</td>
      <td>453467</td>
      <td>45</td>
      <td>1848091</td>
      <td>2011-04-11</td>
      <td>['60-minutes-or-less', 'time-to-make', 'cuisin...</td>
      <td>[595.1, 46.0, 211.0, 22.0, 13.0, 51.0, 26.0]</td>
      <td>12</td>
      <td>['pre-heat oven the 350 degrees f', 'in a mixi...</td>
      <td>this is the recipe that we use at my school ca...</td>
      <td>['white sugar', 'brown sugar', 'salt', 'margar...</td>
      <td>11</td>
      <td>424680.0</td>
      <td>453467.0</td>
      <td>2012-01-26</td>
      <td>5.0</td>
      <td>Originally I was gonna cut the recipe in half ...</td>
    </tr>
  </tbody>
</table>
</div>


### Univariate Analysis

<details>
<summary>Click to View Code</summary>

```python
sns.set_style("whitegrid")
plt.figure(figsize=(15, 12))

# Plot 1: Distribution of Recipe Ratings
plt.subplot(2, 2, 1)
ratings_clean = merged_df['rating'].dropna()
rating_counts = ratings_clean.value_counts().sort_index()
bars = plt.bar(rating_counts.index, rating_counts.values, 
               edgecolor='black', alpha=0.7, color='skyblue', width=0.8)
plt.xlabel('Rating (1-5)')
plt.ylabel('Frequency')
plt.title('Distribution of Recipe Ratings (Discrete)')
plt.xticks([1, 2, 3, 4, 5])

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1000,
             f'{int(height):,}', ha='center', va='bottom')

# Add mean line
plt.axvline(ratings_clean.mean(), color='red', linestyle='dashed', 
           linewidth=2, label=f'Mean: {ratings_clean.mean():.2f}')
plt.legend()

# Plot 2: Distribution of Cooking Time (Minutes)
plt.subplot(2, 2, 2)
cooking_time = merged_df['minutes'].clip(upper=200)  # Cap at 200 min for readibility
plt.hist(cooking_time, bins=30, edgecolor='black', alpha=0.7, color='lightcoral')
plt.xlabel('Cooking Time (Minutes, capped at 200)')
plt.ylabel('Frequency')
plt.title('Distribution of Recipe Cooking Time')
plt.axvline(cooking_time.mean(), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {cooking_time.mean():.2f}')
plt.axvline(cooking_time.median(), color='green', linestyle='dashed', linewidth=2, label=f'Median: {cooking_time.median():.2f}')
plt.legend()

# Plot 3: Distribution of Number of Ingredients
plt.subplot(2, 2, 3)
plt.hist(merged_df['n_ingredients'], bins=20, edgecolor='black', alpha=0.7, color='lightgreen')
plt.xlabel('Number of Ingredients')
plt.ylabel('Frequency')
plt.title('Distribution of Number of Ingredients per Recipe')
plt.axvline(merged_df['n_ingredients'].mean(), color='red', linestyle='dashed', linewidth=2, 
           label=f'Mean: {merged_df["n_ingredients"].mean():.2f}')
plt.axvline(merged_df['n_ingredients'].median(), color='green', linestyle='dashed', linewidth=2, 
           label=f'Median: {merged_df["n_ingredients"].median():.2f}')
plt.legend()

# Plot 4: Distribution of Number of Steps
plt.subplot(2, 2, 4)
plt.hist(merged_df['n_steps'], bins=20, edgecolor='black', alpha=0.7, color='gold')
plt.xlabel('Number of Steps')
plt.ylabel('Frequency')
plt.title('Distribution of Number of Steps per Recipe')
plt.axvline(merged_df['n_steps'].mean(), color='red', linestyle='dashed', linewidth=2, 
           label=f'Mean: {merged_df["n_steps"].mean():.2f}')
plt.axvline(merged_df['n_steps'].median(), color='green', linestyle='dashed', linewidth=2, 
           label=f'Median: {merged_df["n_steps"].median():.2f}')
plt.legend()

plt.tight_layout()
plt.show()

# Summary Statistics
print("UNIVARIATE ANALYSIS - SUMMARY STATISTICS")
numeric_cols = ['minutes', 'n_ingredients', 'n_steps', 'rating']
summary_stats = merged_df[numeric_cols].describe()
print(summary_stats)
```
</details>

    
![png](template_files/template_8_0.png)
    


    UNIVARIATE ANALYSIS - SUMMARY STATISTICS
                minutes  n_ingredients        n_steps         rating
    count  2.344290e+05  234429.000000  234429.000000  219393.000000
    mean   1.067897e+02       9.071506      10.017835       4.679867
    std    3.285975e+03       3.823039       6.442265       0.710471
    min    0.000000e+00       1.000000       1.000000       1.000000
    25%    2.000000e+01       6.000000       6.000000       5.000000
    50%    3.500000e+01       9.000000       9.000000       5.000000
    75%    6.000000e+01      11.000000      13.000000       5.000000
    max    1.051200e+06      37.000000     100.000000       5.000000
    

### Bivariate Analysis

<details>
<summary>Click to View Code</summary>

```python
plt.figure(figsize=(16, 6))

# Plot 1: Cooking Time vs Rating with jittering
plt.subplot(1, 2, 1)
clean_data = merged_df.dropna(subset=['rating'])

# Add jitter to ratings to avoid overplotting
np.random.seed(42)
clean_data['rating_jittered'] = clean_data['rating'] + np.random.uniform(-0.1, 0.1, size=len(clean_data))

# for better visualization
sample_size = min(5000, len(clean_data))
sample_data = clean_data.sample(sample_size, random_state=42)

plt.scatter(sample_data['minutes'].clip(upper=200), sample_data['rating_jittered'], 
           alpha=0.3, s=10, color='blue')
plt.xlabel('Cooking Time (Minutes, capped at 200)')
plt.ylabel('Rating (with jitter)')
plt.title('Cooking Time vs Recipe Rating\n(with jitter to show density)')
plt.grid(True, alpha=0.3)
plt.yticks([1, 2, 3, 4, 5])

# Add average rating line per time bucket
time_bins = [0, 15, 30, 45, 60, 90, 120, 200]
time_labels = [f'{time_bins[i]}-{time_bins[i+1]}min' for i in range(len(time_bins)-1)]
sample_data['time_bin'] = pd.cut(sample_data['minutes'], bins=time_bins, labels=time_labels)
avg_by_bin = sample_data.groupby('time_bin')['rating'].mean()

# Plot average line
for i, (label, avg_rating) in enumerate(avg_by_bin.items()):
    bin_center = (time_bins[i] + time_bins[i+1]) / 2
    plt.plot(bin_center, avg_rating, 'ro', markersize=8)

# Plot 2: Number of Ingredients vs Rating (Violin plot)
plt.subplot(1, 2, 2)
clean_data['ingredient_bins'] = pd.cut(clean_data['n_ingredients'], 
                                       bins=[0, 5, 10, 15, 20, 100], 
                                       labels=['1-5', '6-10', '11-15', '16-20', '20+'])

sns.violinplot(x='ingredient_bins', y='rating', data=clean_data, 
               palette='viridis', inner='quartile')
plt.xlabel('Number of Ingredients (Binned)')
plt.ylabel('Rating')
plt.title('Recipe Rating Distribution by Number of Ingredients')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Correlation matrix
print("BIVARIATE ANALYSIS - CORRELATION MATRIX")
correlation_matrix = merged_df[['minutes', 'n_ingredients', 'n_steps', 'rating']].corr()
print(correlation_matrix)

# Plot correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Heatmap of Key Variables')
plt.tight_layout()
plt.show()
```
</details>
    


    
![png](template_files/template_10_1.png)
    


    BIVARIATE ANALYSIS - CORRELATION MATRIX
                    minutes  n_ingredients   n_steps    rating
    minutes        1.000000      -0.006963  0.011695  0.001440
    n_ingredients -0.006963       1.000000  0.408890 -0.005005
    n_steps        0.011695       0.408890  1.000000 -0.001101
    rating         0.001440      -0.005005 -0.001101  1.000000
    


    
![png](template_files/template_10_3.png)
    


### Interesting Aggregates 

<details>
<summary>Click to View Code</summary>

```python
# Aggregate 1: Average rating by cooking time category
merged_df['cooking_time_category'] = pd.cut(merged_df['minutes'], 
                                           bins=[0, 15, 30, 60, 120, 1000], 
                                           labels=['<15min', '15-30min', '30-60min', '1-2hr', '2hr+'])

agg_time_rating = merged_df.groupby('cooking_time_category').agg({
    'rating': ['mean', 'median', 'count', 'std'],
    'id': 'nunique'  # Count unique recipes
}).round(3)

print("\n1. Average Rating by Cooking Time Category:")
print(agg_time_rating)

# Aggregate 2: Average rating by number of ingredients
merged_df['ingredient_category'] = pd.cut(merged_df['n_ingredients'], 
                                         bins=[0, 5, 10, 15, 20, 100], 
                                         labels=['1-5', '6-10', '11-15', '16-20', '20+'])

agg_ing_rating = merged_df.groupby('ingredient_category').agg({
    'rating': ['mean', 'median', 'count', 'std'],
    'minutes': 'mean'
}).round(3)

print("\n2. Average Rating by Number of Ingredients Category:")
print(agg_ing_rating)

# Aggregate 3: Top performing recipes (by rating count and average)
print("\n3. Top 10 Recipes by Number of Ratings:")
top_recipes = merged_df.groupby('name').agg({
    'rating': ['mean', 'count', 'std'],
    'minutes': 'mean',
    'n_ingredients': 'mean'
}).round(3)
# Filter for recipes with at least 10 ratings
top_recipes_filtered = top_recipes[top_recipes[('rating', 'count')] >= 10]
top_recipes_sorted = top_recipes_filtered.sort_values(by=('rating', 'mean'), ascending=False).head(10)
print(top_recipes_sorted)

# Clean up temporary columns
merged_df.drop(['cooking_time_category', 'ingredient_category'], axis=1, inplace=True, errors='ignore')
if 'ingredient_bins' in clean_data.columns:
    clean_data.drop('ingredient_bins', axis=1, inplace=True)
```
</details>
    

    
    1. Average Rating by Cooking Time Category:
                          rating                           id
                            mean median  count    std nunique
    cooking_time_category                                    
    <15min                 4.719    5.0  46390  0.655   16679
    15-30min               4.681    5.0  55423  0.699   20632
    30-60min               4.668    5.0  65732  0.720   25416
    1-2hr                  4.680    5.0  29757  0.722   12328
    2hr+                   4.623    5.0  20556  0.798    8055
    
    2. Average Rating by Number of Ingredients Category:
                        rating                        minutes
                          mean median   count    std     mean
    ingredient_category                                      
    1-5                  4.712    5.0   39896  0.685  201.223
    6-10                 4.667    5.0  109761  0.725   77.404
    11-15                4.678    5.0   56941  0.703   86.124
    16-20                4.685    5.0   10948  0.706  152.618
    20+                  4.779    5.0    1847  0.619  173.637
    
    3. Top 10 Recipes by Number of Ratings:
                                                       rating            minutes  \
                                                         mean count  std    mean   
    name                                                                           
    a faster egg muffin                                   5.0    12  0.0     2.0   
    2 ingredient punch  mock champagne punch              5.0    11  0.0     5.0   
    2bleu s perfect roast beef                            5.0    10  0.0   605.0   
    zucchini oatmeal bread                                5.0    12  0.0    90.0   
    zesty shrimp salad rolls  rsc                         5.0    10  0.0    12.0   
    yummy onion topping                                   5.0    10  0.0    20.0   
    potato strata with spinach  sausage and goat ch...    5.0    11  0.0    75.0   
    portuguese fried potatoes  batatas a portuguesa       5.0    12  0.0    25.0   
    pork tenderloin with lime and chipotle                5.0    10  0.0   160.0   
    cranberry cashew chocolate bark                       5.0    12  0.0    10.0   
    
                                                       n_ingredients  
                                                                mean  
    name                                                              
    a faster egg muffin                                          4.0  
    2 ingredient punch  mock champagne punch                     6.0  
    2bleu s perfect roast beef                                   2.0  
    zucchini oatmeal bread                                      13.0  
    zesty shrimp salad rolls  rsc                               13.0  
    yummy onion topping                                          6.0  
    potato strata with spinach  sausage and goat ch...          10.0  
    portuguese fried potatoes  batatas a portuguesa              6.0  
    pork tenderloin with lime and chipotle                      11.0  
    cranberry cashew chocolate bark                              4.0  
    
    

## Assessment of Missingness

<details>
<summary>Click to View Code</summary>

```python
sns.set_style("whitegrid")
merged_df['submitted_year'] = pd.to_datetime(merged_df['submitted']).dt.year
missing_mask = merged_df['rating'].isnull()

# Plot 1: Distributions with and without NaN Values

fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left plot: Cooking Time Distribution
cooking_time_capped = merged_df['minutes'].clip(upper=200)

# Histogram for missing ratings
ax1.hist(cooking_time_capped[missing_mask], bins=30, alpha=0.7, 
         color='red', edgecolor='black', label='Rating Missing', density=True)
# Histogram for present ratings
ax1.hist(cooking_time_capped[~missing_mask], bins=30, alpha=0.7, 
         color='blue', edgecolor='black', label='Rating Present', density=True)

ax1.set_xlabel('Cooking Time (minutes, capped at 200)', fontsize=12)
ax1.set_ylabel('Density', fontsize=12)
ax1.set_title('Cooking Time Distribution by Missingness Status', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Right plot: Submission Year Distribution
# Histogram for missing ratings
ax2.hist(merged_df.loc[missing_mask, 'submitted_year'], bins=30, alpha=0.7,
         color='red', edgecolor='black', label='Rating Missing', density=True)
# Histogram for present ratings
ax2.hist(merged_df.loc[~missing_mask, 'submitted_year'], bins=30, alpha=0.7,
         color='blue', edgecolor='black', label='Rating Present', density=True)

ax2.set_xlabel('Submission Year', fontsize=12)
ax2.set_ylabel('Density', fontsize=12)
ax2.set_title('Submission Year Distribution by Missingness Status', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Plot 2: Permutation tests

def run_permutation_test(data, missing_col, test_col, n_permutations=1000):
    test_data = data[[missing_col, test_col]].dropna(subset=[test_col]).copy()
    test_data['missing'] = test_data[missing_col].isnull().astype(int)
    
    # Observed difference
    missing_mean = test_data.loc[test_data['missing'] == 1, test_col].mean()
    present_mean = test_data.loc[test_data['missing'] == 0, test_col].mean()
    observed_diff = abs(missing_mean - present_mean)
    
    # Permutation test
    null_diffs = []
    test_col_values = test_data[test_col].values
    
    for _ in range(n_permutations):
        shuffled_missing = np.random.permutation(test_data['missing'].values)
        shuffled_missing_mean = test_col_values[shuffled_missing == 1].mean()
        shuffled_present_mean = test_col_values[shuffled_missing == 0].mean()
        null_diffs.append(abs(shuffled_missing_mean - shuffled_present_mean))
    
    p_value = np.mean(np.array(null_diffs) >= observed_diff)
    
    return observed_diff, p_value, null_diffs, np.mean(null_diffs)

# Run tests
obs_diff_time, p_time, null_diffs_time, null_mean_time = run_permutation_test(
    merged_df, 'rating', 'minutes', n_permutations=1000
)

obs_diff_year, p_year, null_diffs_year, null_mean_year = run_permutation_test(
    merged_df, 'rating', 'submitted_year', n_permutations=1000
)

# permutation test plots
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 5))

# Left plot: Cooking Time Permutation Test
ax3.hist(null_diffs_time, bins=30, alpha=0.7, color='gray', 
         edgecolor='black', density=True, label='Null Distribution')
ax3.axvline(obs_diff_time, color='red', linewidth=3, 
           label=f'Observed: {obs_diff_time:.2f}')
ax3.axvline(null_mean_time, color='blue', linestyle='--', linewidth=2,
           label=f'Null Mean: {null_mean_time:.2f}')

ax3.set_xlabel('Difference in Mean Cooking Time (minutes)', fontsize=12)
ax3.set_ylabel('Density', fontsize=12)
ax3.set_title(f'Cooking Time Permutation Test\np-value = {p_time:.4f}', 
             fontsize=14, fontweight='bold')
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)

# Right plot: Submission Year Permutation Test
ax4.hist(null_diffs_year, bins=30, alpha=0.7, color='gray', 
         edgecolor='black', density=True, label='Null Distribution')
ax4.axvline(obs_diff_year, color='red', linewidth=3, 
           label=f'Observed: {obs_diff_year:.2f}')
ax4.axvline(null_mean_year, color='blue', linestyle='--', linewidth=2,
           label=f'Null Mean: {null_mean_year:.2f}')

ax4.set_xlabel('Difference in Mean Submission Year', fontsize=12)
ax4.set_ylabel('Density', fontsize=12)
ax4.set_title(f'Submission Year Permutation Test\np-value = {p_year:.4f}', 
             fontsize=14, fontweight='bold')
ax4.legend(fontsize=11)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Clean up temp column
merged_df.drop('submitted_year', axis=1, inplace=True, errors='ignore')

# summary statistics
print("PERMUTATION TEST RESULTS SUMMARY")
print(f"\nCooking Time Test:")
print(f"  Observed difference: {obs_diff_time:.4f} minutes")
print(f"  P-value: {p_time:.4f}")
print(f"  Null mean: {null_mean_time:.4f} minutes")
print(f"  Result: {'FAIL to reject null ' if p_time >= 0.05 else 'REJECT null'}")

print(f"\nSubmission Year Test:")
print(f"  Observed difference: {obs_diff_year:.4f} years")
print(f"  P-value: {p_year:.4f}")
print(f"  Null mean: {null_mean_year:.4f} years")
print(f"  Result: {'FAIL to reject null ' if p_year >= 0.05 else 'REJECT null'}")
```
</details>

    
![png](template_files/template_14_0.png)
    



    
![png](template_files/template_14_1.png)
    


    PERMUTATION TEST RESULTS SUMMARY
    
    Cooking Time Test:
      Observed difference: 51.4524 minutes
      P-value: 0.1210
      Null mean: 18.8967 minutes
      Result: FAIL to reject null (NO DEPENDENCY)
    
    Submission Year Test:
      Observed difference: 0.2963 years
      P-value: 0.0000
      Null mean: 0.0106 years
      Result: REJECT null (DEPENDS)
    

## Hypothesis Testing

Hypothesis Test conducted using the following: 

Null Hypothesis: Cooking time (minutes) and recipe rating are independent. 

Alternate Hypothesis: Cooking time and recipe rating are correlated

Test Statistic: Pearson's Correlation Coefficient (r)

<details>
<summary>Click to View Code</summary>

```python
clean_data = merged_df.dropna(subset=['minutes', 'rating'])

x = clean_data['minutes'].to_numpy()
y = clean_data['rating'].to_numpy()

n = len(x)

# Observed test statistic
observed_r = np.corrcoef(x, y)[0, 1]

# Standardize x once (important speedup)
x_std = (x - x.mean()) / x.std()

n_permutations = 10_000
null_stats = np.empty(n_permutations)

# Permutation test
for i in range(n_permutations):
    y_perm = np.random.permutation(y)
    y_perm_std = (y_perm - y.mean()) / y.std()
    null_stats[i] = (x_std @ y_perm_std) / n

# Two-sided p-value
p_value = np.mean(np.abs(null_stats) >= np.abs(observed_r))

# Results
print("Permutation Test: Cooking Time vs Recipe Rating")
print("-" * 50)
print(f"Sample size: {n:,}")
print(f"Observed correlation (r): {observed_r:.6f}")
print(f"P-value: {p_value:.6f}")
print(f"Significant at alpha = 0.05? {'YES' if p_value < 0.05 else 'NO'}")

```
</details>

    Permutation Test: Cooking Time vs Recipe Rating
    Sample size: 219,393
    Observed correlation (r): 0.001440
    P-value: 0.455600
    Significant at alpha = 0.05? NO
    

## Framing a Prediction Problem

Model will aim to predict ratings of recipes.

## Baseline Model

<details>
<summary>Click to View Code</summary>

```python
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Prepping data

data = merged_df.dropna(subset=['rating']).copy()

# Extract first tag 
def extract_first_tag(tags):
    if pd.isna(tags):
        return 'missing'
    if isinstance(tags, str):
        # Remove brackets and split
        clean_tags = tags.strip("[]'").replace("'", "")
        tags_list = [t.strip() for t in clean_tags.split(',')]
        return tags_list[0] if tags_list else 'no-tag'
    return 'invalid'

data['first_tag'] = data['tags'].apply(extract_first_tag)

# Only keep top 10 most common tags
top_tags = data['first_tag'].value_counts().head(10).index
data['first_tag'] = data['first_tag'].apply(lambda x: x if x in top_tags else 'other')

data = data.dropna(subset=['minutes', 'n_ingredients'])

# features and target
X = data[['minutes', 'n_ingredients', 'first_tag']]
y = data['rating']

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=data['first_tag']
)

print("=" * 60)
print("BASELINE MODEL (FIXED)")
print("=" * 60)
print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")
print(f"Unique tags after limiting: {X_train['first_tag'].nunique()}")

# Create Pipeline

numerical_features = ['minutes', 'n_ingredients']
categorical_features = ['first_tag']

# Numerical transformer (scaling)
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Categorical transformer with limited categories
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(categories='auto', 
                            handle_unknown='ignore', 
                            sparse_output=False,
                            drop='first')) 
])

# preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Use Ridge regression (regularized)
baseline_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', Ridge(alpha=1.0))  # Regularization prevents overfitting
])

# Training and eval

print("\nTraining model with regularization...")
baseline_pipeline.fit(X_train, y_train)

# Predictions
y_train_pred = baseline_pipeline.predict(X_train)
y_test_pred = baseline_pipeline.predict(X_test)

# Metrics
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Print results

print("\n" + "=" * 60)
print("MODEL PERFORMANCE (FIXED)")
print("=" * 60)
print(f"{'Metric':<15} {'Training':<12} {'Test':<12}")
print(f"{'-'*15} {'-'*12} {'-'*12}")
print(f"{'RMSE':<15} {train_rmse:<12.4f} {test_rmse:<12.4f}")
print(f"{'MAE':<15} {train_mae:<12.4f} {test_mae:<12.4f}")
print(f"{'R squared':<15} {train_r2:<12.4f} {test_r2:<12.4f}")


# Summary

print("\n" + "=" * 60)
print("MODEL DIAGNOSTICS")
print("=" * 60)
print(f"Features after encoding: {len(all_feature_names)}")
print(f"Ridge regularization (alpha): {baseline_pipeline.named_steps['regressor'].alpha}")
print(f"Training vs test RMSE ratio: {train_rmse/test_rmse:.3f} (should be close to 1)")

print("\nTop 5 feature coefficients:")
print(feature_importance.head(5).to_string(index=False))
```
</details>

    ============================================================
    BASELINE MODEL (FIXED)
    ============================================================
    Training samples: 175514
    Test samples: 43879
    Unique tags after limiting: 11
    
    Training model with regularization...
    
    ============================================================
    MODEL PERFORMANCE (FIXED)
    ============================================================
    Metric          Training     Test        
    --------------- ------------ ------------
    RMSE            0.7096       0.7099      
    MAE             0.4941       0.4936      
    R squared       0.0021       0.0029      
    
    ============================================================
    MODEL DIAGNOSTICS
    ============================================================
    Features after encoding: 12
    Ridge regularization (alpha): 1.0
    Training vs test RMSE ratio: 1.000 (should be close to 1)
    
    Top 5 feature coefficients:
                         Feature  Coefficient
                first_tag_course    -0.127182
               first_tag_curries    -0.103545
               first_tag_lactose    -0.091713
    first_tag_60-minutes-or-less    -0.054053
                 first_tag_other    -0.046938
    

## Final Model

<details>
<summary>Click to View Code</summary>

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# quick data prep (same as baseline)
data = merged_df.dropna(subset=['rating']).copy()

def get_first_tag(tags):
    if pd.isna(tags) or not isinstance(tags, str):
        return 'other'
    return tags.split(',')[0].strip("[]' ")[:20]

data['first_tag'] = data['tags'].apply(get_first_tag)
top_tags = data['first_tag'].value_counts().head(3).index
data['first_tag'] = data['first_tag'].apply(lambda x: x if x in top_tags else 'other')

# Engineer 2 new features
data['steps_per_ingredient'] = data['n_steps'] / (data['n_ingredients'].clip(1))
data['log_minutes'] = np.log1p(data['minutes'])

data = data.dropna(subset=['minutes', 'n_ingredients', 'steps_per_ingredient', 'log_minutes'])

# Features
X = data[['minutes', 'n_ingredients', 'first_tag', 'steps_per_ingredient', 'log_minutes']]
y = data['rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Split: 80, 20")

# Pipeline with hyper parameter plan

print("\n" + "="*50)
print("HYPERPARAMETER TUNING PLAN")
print("="*50)
print("Model: RandomForestRegressor")
print("Parameters to tune:")
print("- n_estimators: [30, 50] (trees in forest)")
print("- max_depth: [5, 8] (tree depth)")
print("- min_samples_split: [5, 10] (samples to split)")

numerical_cols = ['minutes', 'n_ingredients', 'steps_per_ingredient', 'log_minutes']
categorical_cols = ['first_tag']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=42))  
])

# minimal parameter grid
param_grid = {
    'model__n_estimators': [30, 50],
    'model__max_depth': [5, 8],
    'model__min_samples_split': [5, 10]
}

# Hyperparameter tuning

print("\nRunning quick GridSearchCV (2x2x2 = 8 fits)...")

grid_search = GridSearchCV(
    pipeline, 
    param_grid, 
    cv=2,  
    scoring='neg_mean_squared_error',
    n_jobs=1,  
    verbose=1  
)

grid_search.fit(X_train, y_train)
print(f"Best params: {grid_search.best_params_}")

# final model
final_model = grid_search.best_estimator_

# eval and comparisons
y_pred = final_model.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
test_r2 = r2_score(y_test, y_pred)

# Baseline results
baseline_r2 = 0.0029
baseline_rmse = 0.7099

print("\n" + "="*50)
print("RESULTS")
print("="*50)
print(f"{'Metric':<10} {'Baseline':<10} {'Final':<10} {'Change':<10}")
print(f"{'-'*10} {'-'*10} {'-'*10} {'-'*10}")
print(f"{'R squared':<10} {baseline_r2:<10.4f} {test_r2:<10.4f} {test_r2-baseline_r2:>+9.4f}")
print(f"{'RMSE':<10} {baseline_rmse:<10.4f} {test_rmse:<10.4f} {baseline_rmse-test_rmse:>+9.4f}")

# Plot: Feature importance
rf = final_model.named_steps['model']
importances = rf.feature_importances_

cat_encoder = final_model.named_steps['preprocessor'].named_transformers_['cat']
cat_names = cat_encoder.get_feature_names_out(['first_tag'])
all_names = numerical_cols + list(cat_names)

# Top 5 features
top_idx = np.argsort(importances)[-5:]
top_names = [all_names[i] for i in top_idx]
top_imp = [importances[i] for i in top_idx]

fig, ax = plt.subplots(figsize=(8, 5))

# horizontal bar plot
bars = ax.barh(range(len(top_imp)), top_imp)
ax.set_yticks(range(len(top_imp)))
ax.set_yticklabels(top_names, fontsize=10)
ax.set_xlabel('Importance', fontsize=12)
ax.set_title('Top 5 Feature Importances (Final Model)', fontsize=14, fontweight='bold')
ax.grid(alpha=0.3, axis='x')

# add value labels on bars
for i, (bar, imp) in enumerate(zip(bars, top_imp)):
    width = bar.get_width()
    ax.text(width + 0.001, bar.get_y() + bar.get_height()/2,
            f'{imp:.3f}', ha='left', va='center', fontsize=9)

plt.tight_layout()
plt.show()
```
</details>

    Split: 80, 20
    
    ==================================================
    HYPERPARAMETER TUNING PLAN
    ==================================================
    Model: RandomForestRegressor
    Parameters to tune:
    - n_estimators: [30, 50] (trees in forest)
    - max_depth: [5, 8] (tree depth)
    - min_samples_split: [5, 10] (samples to split)
    
    Running quick GridSearchCV (2x2x2 = 8 fits)...
    Fitting 2 folds for each of 8 candidates, totalling 16 fits
    Best params: {'model__max_depth': 8, 'model__min_samples_split': 10, 'model__n_estimators': 30}
    
    ==================================================
    RESULTS
    ==================================================
    Metric     Baseline   Final      Change    
    ---------- ---------- ---------- ----------
    R²         0.0029     0.0063       +0.0034
    RMSE       0.7099     0.7116       -0.0017
    


    
![png](template_files/template_23_1.png)
    


## Fairness Analysis

Hypothesis Test conducted using the following: 

Null Hypothesis: The model is fair with respect to cook time. 
Any difference in RMSE between recipes that take less than 30 minutes 
and those that take 30 minutes or more is due to random chance.

Alternate Hypothesis: The model is unfair with respect to cook time.
The model has higher RMSE for recipes that take less than 30 minutes
than for recipes that take 30 minutes or more.

Test Statistic: difference in RMSE

<details>
<summary>Click to View Code</summary>

```python
y_pred = final_model.predict(X_test)

short_mask = X_test['minutes'] < 30
long_mask  = X_test['minutes'] >= 30

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

rmse_short = rmse(y_test[short_mask], y_pred[short_mask])
rmse_long  = rmse(y_test[long_mask],  y_pred[long_mask])

observed_stat = rmse_short - rmse_long

n_permutations = 10_000
null_stats = np.empty(n_permutations)

# Convert masks to array for permutation
group_labels = short_mask.to_numpy()

for i in range(n_permutations):
    permuted_labels = np.random.permutation(group_labels)

    rmse_short_perm = rmse(
        y_test[permuted_labels],
        y_pred[permuted_labels]
    )
    
    rmse_long_perm = rmse(
        y_test[~permuted_labels],
        y_pred[~permuted_labels]
    )

    null_stats[i] = rmse_short_perm - rmse_long_perm

p_value = np.mean(null_stats >= observed_stat)

print("Fairness Analysis: Cook Time < 30 vs ≥ 30 Minutes")
print("-" * 60)
print(f"RMSE (short recipes): {rmse_short:.4f}")
print(f"RMSE (long recipes):  {rmse_long:.4f}")
print(f"Observed RMSE difference (short − long): {observed_stat:.4f}")
print(f"P-value (permutation test): {p_value:.6f}")

```
</details>

    Fairness Analysis: Cook Time < 30 vs ≥ 30 Minutes
    ------------------------------------------------------------
    RMSE (short recipes): 0.6744
    RMSE (long recipes):  0.7333
    Observed RMSE difference (short − long): -0.0589
    P-value (permutation test): 1.000000
    

