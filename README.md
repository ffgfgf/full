Модуль 1 
1.1 Парсинг данных
#Импортирование необходимых библиотек
import pandas as pd
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
import seaborn as sns
​
import codecs
import json
import glob
pd.set_option('display.max_columns', None)
from pandas import json_normalize
Импорт статей

#Путь к файлам .geojson
​
path = 'C:/dark/ASAS/AFAF/Data'
file = glob.glob(path + "/*.json")
df_full=pd.DataFrame()
​
#df_full=pd.DataFrame(columns=['Пост', 'день публикации', 'месяц публикации', 'время публикации'])
#Цикл для получения файла и его загрузки, используя json.load 
for filename in file:
    name = filename.split("\\")[-1][:-5]
    with codecs.open(filename, 'r', 'utf-8-sig') as json_file:  
        data = json.load(json_file)
    
    for article in data['refs']:  
        if article != None:
            df=pd.concat(
                [
                    pd.DataFrame([article[0]],columns=['Post']),
                    json_normalize(article[1]),
                    pd.DataFrame([name],columns=['Company'])
                ],
                axis=1
            )
            df_full=pd.concat([df_full,df],axis=0,ignore_index=True)
df_full.info()


df_full.head()


#Путь к файлам .geojson
path = 'C:/dark/ASAS/AFAF/Data'
file = glob.glob(path + "/*.json")


df = pd.DataFrame(columns=['rate','subs','industries','about','Company']) 

#Цикл для получения файла и его загрузки, используя json.load 
for filename in file:
    with codecs.open(filename, 'r', 'utf-8-sig') as json_file:  
        data = json.load(json_file)
        name = filename.split("\\")[-1][:-5]   
        try:
            company_info=pd.concat([json_normalize(data['info']),pd.DataFrame([name],columns=['Company'])],axis=1)
        except:
            d={'rate':['Не указано'],'subs':['Не указано'],'industries':['Не указано'],'about':['Не указано']}
            company_info=pd.concat([pd.DataFrame(d),pd.DataFrame([name],columns=['Company'])],axis=1)
    df = pd.concat([df,company_info], axis=0, ignore_index=True)
df.head()


tk = df_full.merge(df, on='Company',how='left')
tk.shape


tk.head(200)

1.3 Предварительная обработка данных

import pymorphy2
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')

nltk.download('punkt')

sw = stopwords.words('russian')
morph = pymorphy2.MorphAnalyzer()

def clear_text(text):
    text=text.lower()
    text = re.sub(r'[^а-яё ]','', str(text))
    tokens=word_tokenize(text, language="russian")
    tokens = [morph.parse(i)[0].normal_form for i in tokens]
    tokens = [i for i in tokens if i not in sw and len(i) > 3]
    return tokens
    
tk['lemmatize_tokens'] = tk['Post'].apply(clear_text)

tk.head()

tk['clear_text'] = tk['lemmatize_tokens'].apply(lambda x: " ".join(x))

tk.head()

tk.to_csv('data.csv', index=False, encoding='utf-8-sig')

tk.info()

1.4 Векторизация текста и поиск ngram
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer

tfidf = TfidfVectorizer(min_df=5,max_df=0.8, ngram_range=(1, 2))
X_tfidf = tfidf.fit_transform(tk['clear_text'])

df_tfidf = pd.DataFrame(X_tfidf.toarray(), columns = tfidf.get_feature_names_out())
df_tfidf.head()

X_tfidf

df_tfidf["Company"]=tk["Company"]

count = CountVectorizer(min_df=5,max_df=0.8, ngram_range=(1, 2))
X_count = count.fit_transform(tk['clear_text'])
df_count = pd.DataFrame(X_count.toarray(),columns = count.get_feature_names_out())
df_count.head()

X_count

df_count["Company"]=tk["Company"]

1.6 Кластеризация
from sklearn.cluster import KMeans, Birch, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture

X = df_tfidf.drop(["Company"], axis=1)

#обьявляю MinMaxScaler
scaler = MinMaxScaler()
#преобразую данные
X = scaler.fit_transform(X)

#Обьявляю метод главных компонент
pca = PCA(n_components=2)
#Применяю его на данных
X = pca.fit_transform(X)

#Функция для визуализации распределения 
def viz(prediction):
    #Размер фигуры
    plt.figure(figsize=(12, 12))
    plt.subplot(224)
    #Выводить изображение буду при помощи scatter
    plt.scatter(X[:, 0], X[:, 1], c=prediction)
    plt.title("Unevenly Sized Blobs")
    #Вывод изображения
    plt.show()
    
    #Настройка параметров
kmeans = KMeans(n_clusters=4, random_state=0).fit(X)
#Предсказание
kmpreds = kmeans.predict(X)
#Заношу кластеризированные метки в набор данных
df_tfidf["KMCLUSTS"] = kmpreds

#Применяю функцию описаную выше
viz(df_tfidf["KMCLUSTS"])

#Настройка параметров
kbmeans = MiniBatchKMeans(n_clusters=4, random_state=0, batch_size=4096)
#Заношу кластеризированные метки в набор данных
df_tfidf["KBCLUSTS"] = kbmeans.fit_predict(X)

viz(df_tfidf["KBCLUSTS"])

#Настройка параметров
gm = GaussianMixture(n_components=2, random_state=0).fit_predict(X)
#Заношу кластеризированные метки в набор данных
df_tfidf["GMCLUSTS"] = gm

viz(df_tfidf["GMCLUSTS"])

2.2 Классификация
df=pd.read_json("C:/dark/ASAS/AFAF/Target.json")
df = df.rename(columns = {"Сompany": "Company"})
df.head()

df_tfidf["Company"]

df_tfidf=df_tfidf.merge(df, on='Company')
df_tfidf['Nominations']

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(20,14))
sns.histplot(data=df_tfidf,x='Nominations')

from sklearn.model_selection import train_test_split

x=df_tfidf.drop(['Nominations', "Company"], axis=1)
y=df_tfidf['Nominations']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42, stratify=y)

from sklearn.tree import DecisionTreeClassifier as Tree

tree = Tree(max_depth=20, min_samples_split=4, min_samples_leaf=2)

# Выбираем все столбцы, содержащие строки
string_columns = x_train.select_dtypes(include='object').columns

# Применяем One-Hot Encoding ко всем строковым столбцам
x_train = pd.get_dummies(x_train, columns=string_columns)


tree.fit(x_train, y_train)

Оценка модели
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import learning_curve
import numpy as np

# Разделение данных на обучающий и тестовый наборы
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42, stratify=y)


# Выбор значимых признаков
sel = SelectFromModel(Tree(max_depth=20, min_samples_split=4, min_samples_leaf=2))
sel.fit(x_train, y_train)


# Понижение размерности
x_train_selected = sel.transform(x_train)
x_test_selected = sel.transform(x_test)

# Настройка гиперпараметров
param_grid = {'max_depth': [10, 20, 30, 40],
              'min_samples_split': [2, 4, 6, 8],
              'min_samples_leaf': [1, 2, 3, 4]}

grid_search = GridSearchCV(Tree(), param_grid, cv=5)
grid_search.fit(x_train_selected, y_train)


# Вывод наилучших параметров
print("Наилучшие параметры: ", grid_search.best_params_)

# Оценка качества полученной модели
y_pred = grid_search.predict(x_test_selected)
print(classification_report(y_test, y_pred))

Визуализация дерева
from sklearn.metrics import confusion_matrix
import seaborn as sns
​
# Assuming you have y_test and y_pred variables representing the true labels and predicted labels, respectively
​
# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)
​
# Plot the confusion matrix using seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

print(classification_report(y_test,y_pred))

from sklearn.ensemble import RandomForestClassifier

# Assuming you have already trained a RandomForestClassifier and assigned it to the 'model' variable

# Access the feature importances
importances = model.feature_importances_

# Print the feature importances
for feature_name, importance in zip(x_train.columns, importances):
    print(f"{feature_name}: {importance}")


model.feature_importances_

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Assuming you have already trained a RandomForestClassifier and assigned it to the 'model' variable
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Access the feature importances
importances = model.feature_importances_
feature_names = x_train.columns

# Create DataFrame of feature importances
importance_df = pd.DataFrame({'Feature Importance': importances}, index=feature_names)

print(importance_df)


from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree



# Assuming you have already trained a RandomForestClassifier and assigned it to the 'model' variable
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Select an individual tree from the random forest
tree_index = 0 

# Visualize the selected tree
plt.figure(figsize=(60, 30),dpi=150)
plot_tree(model.estimators_[tree_index], filled=True, rounded=True, feature_names=x_train.columns)
plt.show()
