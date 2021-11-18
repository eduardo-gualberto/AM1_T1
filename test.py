from model import decision_tree, test_df
import warnings
warnings.filterwarnings("ignore")

dt = decision_tree
# atributos para test, exc. 'class'
X_t = test_df[test_df.columns[:-1]]

# atributo classe para rotulação
Y_t = test_df['class']

div_count = 0

FP = 0
TP = 0
FN = 0
TN = 0

for i in range(len(X_t.index)):
    predicted = dt.predict([X_t.iloc[i]])[0]
    truth = Y_t.iloc[i]
    if predicted != truth:
        div_count += 1
    if predicted == 1 and truth == 1:
        TP += 1
    elif predicted == 1 and truth == 0:
        FP += 1
    elif predicted == 0 and truth == 1:
        FN += 1
    elif predicted == 0 and truth == 0:
        TN += 1
print(f"\nTP = {TP}, FP = {FP}, FN = {FN}, TN = {TN}\nTotal Divergencias = {div_count}\n")
