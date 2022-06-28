import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv(r"C:\Users\Guyzaid\PycharmProjects\ml_hw4\all.csv")
# df = df[df['Measure type'] == 'roc_auc_score']

# sns.catplot(
#     data=df,
#     x='Number of features selected', y='Measure value', col='Measure type', row='Numer of samples',
#     kind='bar', hue='Learning algorithm'
# )

# sns.catplot(
#     data=df,
#     x='Number of features selected', y='Measure value', col='Measure type', row='Numer of samples',
#     kind='bar', hue='Filtering algorithm'
# )

# sns.catplot(
#     data=df,
#     x='Filtering algorithm', y='Feature selection time (ms)',
#     col='original number of features',
#     # kind='bar', hue='Filtering algorithm'
# )
#
# plt.show()

b = df.groupby(['Dataset Name', 'Filtering algorithm', 'Measure type'])['Measure value'].mean()

from scipy import stats

# preparing data for friedman test
mrmr = b.iloc[0::4].values
select_fdr = b.iloc[1::4].values
f_class = b.iloc[2::4].values
pca_ig = b.iloc[3::4].values

# perform Friedman Test
x = stats.friedmanchisquare(mrmr, select_fdr, f_class, pca_ig)
print(x)
