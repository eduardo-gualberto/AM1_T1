import matplotlib.pyplot as plt
import graphviz



from sklearn import tree
import pandas as pd

# dataframe inteiro baixado do OpenML
# https://www.openml.org/d/44
raw_df = pd.read_csv('dataset_44_spambase.csv')
# declaração de atributos de interesse
raw_df = raw_df[['word_freq_make', 'word_freq_address', 'word_freq_all',
                 'word_freq_our', 'word_freq_over', 'word_freq_remove',
                 'word_freq_internet', 'word_freq_order', 'word_freq_mail',
                 'word_freq_receive', 'word_freq_will', 'word_freq_people',
                 'word_freq_report', 'word_freq_addresses', 'word_freq_free',
                 'word_freq_business', 'word_freq_email', 'word_freq_you',
                 'word_freq_credit', 'word_freq_your', 'word_freq_font',
                 'word_freq_money',  'word_freq_george',
                 'word_freq_lab', 'word_freq_labs', 'word_freq_telnet',
                 'word_freq_data',
                 'word_freq_technology', 'word_freq_parts',
                 'word_freq_direct', 'word_freq_meeting',
                 'word_freq_original', 'word_freq_project',
                 'word_freq_edu', 'word_freq_table', 'word_freq_conference', 'capital_run_length_average',
                 'capital_run_length_longest', 'capital_run_length_total', 'class']]


# criação dos dados de treino para 30% do numero total de registros
train_df = raw_df.sample(frac=.7, replace=True, random_state=1)

# criação dos dados de teste para 70% do numero total de registros
test_df = raw_df.sample(frac=.3, replace=True, random_state=1)

# remover atributo classe da lista de atributos
# o atributo classe é o último atributo
credit_df = train_df[train_df.columns[:-1]]
credit_target_df = train_df['class']

# X = atributos que vao participar das decisões
X = credit_df
# y = lista de classes atribuidas para cada registro do dataframe original
Y = credit_target_df


# garantir que em todos os nós folha existam ao menos 5 exemplos
decision_tree = tree.DecisionTreeClassifier(min_samples_leaf=5)
decision_tree = decision_tree.fit(X, Y)


# dot_data = tree.export_graphviz(dt_setup(), feature_names=credit_df.columns, class_names=[
#                                 "Spam" if x == 0 else "Ham" for x in Y.tolist()], filled=True)
# graph = graphviz.Source(dot_data)
# graph.render(view=True)


# FAZER PARTE PARA OS TESTES