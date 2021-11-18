import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

df = pd.read_csv('dataset_44_spambase.csv')
print(df.columns)
df = df[['char_freq_%3B', 'char_freq_%28', 'char_freq_%5B',
         'char_freq_%24', 'class']]

sb.pairplot(df, hue='class')
plt.show()
