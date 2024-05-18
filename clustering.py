import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colors
from matplotlib.colors import ListedColormap
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.cm as cm
import matplotlib.pyplot as plt, numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering
from datetime import datetime
import matplotlib.pyplot as plt

data = pd.read_csv("marketing_campaign.xls",sep='\t')
data.Dt_Customer = pd.to_datetime(data.Dt_Customer,format="%d-%m-%Y")
data.insert(2,"Age",data.Dt_Customer.dt.year-data.Year_Birth)
data["Marital_Status"] = data["Marital_Status"].replace({"Married":"Partner", "Together":"Partner", "Absurd":"Alone", "Widow":"Alone", "YOLO":"Alone", "Divorced":"Alone", "Single":"Alone",})
data = data.dropna()
data.insert(6,"Spent",data["MntWines"]+ data["MntFruits"]+ data["MntMeatProducts"]+ data["MntFishProducts"]+ data["MntSweetProducts"]+ data["MntGoldProds"])
#data.head()

plt.figure(figsize=(15, 9))

histogram = sns.histplot(
    data.Age, 
    bins=50, 
    stat='density',
    alpha=0.2,
    color = "red"
)

density_curve = sns.kdeplot(data.Age, linewidth=3, color="red")

plt.xlabel("Età", labelpad=20)
plt.ylabel("Densità", labelpad=20)
plt.show()

data = data[data.Age < 100]

mean_age = data.Age.mean()
median_age = data.Age.median()
std_age = data.Age.std()
print("Media: " + str(mean_age))
print("Mediana: " + str(median_age))
print("Standard deviation: " + str(std_age))

plt.figure(figsize=(15, 9))

histogram = sns.histplot(
    data.Income, 
    bins=50, 
    stat='density',
    alpha=0.5,
    color = "blue"
)

density_curve = sns.kdeplot(data.Income, linewidth=3, color="grey")

plt.xlabel("Stipendio", labelpad=20)
plt.ylabel("Densità", labelpad=20)
plt.show()

data = data[data.Income < 600000]

mean_income = data.Income.mean()
median_income = data.Income.median()
std_income = data.Income.std()
print("Media: " + str(mean_income))
print("Mediana: " + str(median_income))
print("Standard deviation: " + str(std_income))

counts_edu = data.Education.value_counts()
#print(counts_edu)

total = counts_edu.sum()

# Funzione per mostrare sia i conteggi che le percentuali
def autopct(pct):
    count = int(round(pct * total / 100))
    return f'{count}\n({pct:.1f}%)'
plt.figure(figsize=(15,9))
plt.pie(counts_edu, labels=counts_edu.index, autopct=autopct, startangle=90)
plt.axis('equal')
plt.title('Distribuzione dei livelli di istruzione')
plt.show()

counts_status = data.Marital_Status.value_counts()
#print(counts_status)

total = counts_status.sum()

# Funzione per mostrare sia i conteggi che le percentuali
def autopct(pct):
    count = int(round(pct * total / 100))
    return f'{count}\n({pct:.1f}%)'
plt.figure(figsize=(15,9))
plt.pie(counts_status, labels=counts_status.index, autopct=autopct, startangle=90)
plt.axis('equal')
plt.title('Distribuzione stato civile')
plt.show()

counts_children = data.Kidhome.value_counts()
#print(counts_children)

total = counts_children.sum()

# Funzione per mostrare sia i conteggi che le percentuali
def autopct(pct):
    count = int(round(pct * total / 100))
    return f'{count}\n({pct:.1f}%)'
plt.figure(figsize=(15,9))
plt.pie(counts_children, labels=counts_children.index, autopct=autopct, startangle=90)
plt.axis('equal')
plt.title('Bambini')
plt.show()

counts_teen = data.Teenhome.value_counts()
#print(counts_teen)

total = counts_teen.sum()

# Funzione per mostrare sia i conteggi che le percentuali
def autopct(pct):
    count = int(round(pct * total / 100))
    return f'{count}\n({pct:.1f}%)'
plt.figure(figsize=(15,9))
plt.pie(counts_teen, labels=counts_teen.index, autopct=autopct, startangle=90)
plt.axis('equal')
plt.title('Adolescenti')
plt.show()

data.insert(9,"Children",data["Kidhome"]+ data["Teenhome"])
#data.head()

counts_figli = data.Children.value_counts()
#print(counts_figli)

total = counts_figli.sum()

# Funzione per mostrare sia i conteggi che le percentuali
def autopct(pct):
    count = int(round(pct * total / 100))
    return f'{count}\n({pct:.1f}%)'
plt.figure(figsize=(15,9))
plt.pie(counts_figli, labels=counts_figli.index, autopct=autopct, startangle=90)
plt.axis('equal')
plt.title('Figli')
plt.show()

phd_data = data.loc[data.Education== 'PhD']
#phd_data.head()

plt.figure(figsize=(15, 9))

histogram = sns.histplot(
    phd_data.Income, 
    bins=20, 
    stat='density',
    alpha=1,
    color = "orange"
)

density_curve = sns.kdeplot(phd_data.Income, linewidth=3, color="red")

plt.xlabel("Income", labelpad=20)
plt.ylabel("Density", labelpad=20)
plt.title("Phd income")
#plt.show()

mean_income = phd_data.Income.mean()
median_income = phd_data.Income.median()
std_income = phd_data.Income.std()
print("Media: " + str(mean_income))
print("Mediana: " + str(median_income))
print("Standard deviation: " + str(std_income))

basic_data = data.loc[data.Education== 'Basic']
#basic_data.head()

plt.figure(figsize=(15, 9))

histogram = sns.histplot(
    basic_data.Income, 
    bins=20, 
    stat='density',
    alpha=0.5,
    color = "green"
)

density_curve = sns.kdeplot(basic_data.Income, linewidth=3, color="green")

plt.title("Basic Education Income")
plt.xlabel("Income", labelpad=20)
plt.ylabel("Density", labelpad=20)
#plt.show()

mean_income = basic_data.Income.mean()
median_income = basic_data.Income.median()
std_income = basic_data.Income.std()
print("Media: " + str(mean_income))
print("Mediana: " + str(median_income))
print("Standard deviation: " + str(std_income))

graduation_data = data.loc[data.Education== 'Graduation']
#graduation_data.head()

plt.figure(figsize=(15, 9))

histogram = sns.histplot(
    graduation_data.Income, 
    bins=20, 
    stat='density',
    alpha=0.5,
    color = "violet"
)

density_curve = sns.kdeplot(graduation_data.Income, linewidth=3, color="violet")

plt.title("Graduation income")
plt.xlabel("Income", labelpad=20)
plt.ylabel("Density", labelpad=20)
#plt.show()

graduation_data = graduation_data[graduation_data.Income < 600000]

mean_income = graduation_data.Income.mean()
median_income = graduation_data.Income.median()
std_income = graduation_data.Income.std()
print("Media: " + str(mean_income))
print("Mediana: " + str(median_income))
print("Standard deviation: " + str(std_income))

cycle_data = data.loc[data.Education== '2n Cycle']
#cycle_data.head()

plt.figure(figsize=(15, 9))

histogram = sns.histplot(
    cycle_data.Income, 
    bins=20, 
    stat='density',
    alpha=0.7,
    color = "blue"
)

density_curve = sns.kdeplot(cycle_data.Income, linewidth=3, color="blue")

plt.title("2dn cycle Income")
plt.xlabel("Income", labelpad=20)
plt.ylabel("Density", labelpad=20)
#plt.show()

mean_income = cycle_data.Income.mean()
median_income = cycle_data.Income.median()
std_income = cycle_data.Income.std()
print("Media: " + str(mean_income))
print("Mediana: " + str(median_income))
print("Standard deviation: " + str(std_income))

master_data = data.loc[data.Education== 'Master']
#master_data.head()

plt.figure(figsize=(15, 9))

histogram = sns.histplot(
    master_data.Income, 
    bins=20, 
    stat='density',
    alpha=0.5,
    color = "green"
)

density_curve = sns.kdeplot(master_data.Income, linewidth=3, color="green")

plt.title("Master income")
plt.xlabel("Income", labelpad=20)
plt.ylabel("Density", labelpad=20)
#plt.show()

mean_income = master_data.Income.mean()
median_income = master_data.Income.median()
std_income = master_data.Income.std()
print("Media: " + str(mean_income))
print("Mediana: " + str(median_income))
print("Standard deviation: " + str(std_income))

plt.figure(figsize=(15,10))

density_curve = sns.kdeplot(phd_data.Income, linewidth=3, color="green")
density_curve = sns.kdeplot(master_data.Income, linewidth=3, color="orange")
density_curve = sns.kdeplot(cycle_data.Income, linewidth=3, color="red")
density_curve = sns.kdeplot(graduation_data.Income, linewidth=3, color="blue")
density_curve = sns.kdeplot(basic_data.Income, linewidth=3, color="violet")

plt.title("Distribuzione stipendio in base al livello di educazione")
plt.xlabel("Stipendio", labelpad=20)
plt.ylabel("Densità", labelpad=20)
plt.legend()
plt.show()

s = (data.dtypes == 'object')
object_cols = list(s[s].index)

print("Categorical variables in the dataset:", object_cols)

LE=LabelEncoder()
for i in object_cols:
    data[i]=data[[i]].apply(LE.fit_transform)
    
print("All features are now numerical")

#heatmap per vedere correlazione tra gli attributi
data.drop(["ID","Year_Birth"],axis=1, inplace=True)
data.drop(data.iloc[:, 16:], inplace = True, axis = 1)
data=data.select_dtypes(include=['number'])
plt.figure(figsize=(15,10))
 
ax = sns.heatmap(data.corr(), annot=True)

#grafico pairplot per vedere correlazione tra alcuni attributi e il numero dei figli dei clienti
sns.set_theme(rc={"axes.facecolor":"#FFF9ED","figure.facecolor":"#FFF9ED"})
pallet = ["#682F2F", "#B9C0C9", "#9F8A78", "#F3AB60"]
cmap = colors.ListedColormap(["#682F2F", "#B9C0C9", "#9F8A78", "#F3AB60"])
To_Plot = [ "Income", "Age", "Spent", "Children"]
print("Reletive Plot Of Some Selected Features: A Data Subset")
plt.figure()
sns.pairplot(data[To_Plot], hue= "Children",palette= (["r","b","y","m"])) 
plt.show()

data = data.drop(["Teenhome","Kidhome"],axis=1)
data.head()

scaler = StandardScaler()
scaler.fit(data)
scaled_data = pd.DataFrame(scaler.transform(data),columns=data.columns)

#PCA per ridurre a 3 dimensioni
pca = PCA(n_components=3)
pca.fit(scaled_data)
PCA_ds = pd.DataFrame(pca.transform(scaled_data), columns=(["col1","col2", "col3"]))
PCA_ds.describe().T

x =PCA_ds["col1"]
y =PCA_ds["col2"]
z =PCA_ds["col3"]

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(x,y,z, marker="o", c=z)
ax.set_title("Proiezione nello spazio 3D")
plt.show()

# Clustering
#numero di cluster da utilizzare usando il metodo a gomito con PCA
print('Elbow Method to determine the number of clusters to be formed:')
Elbow_M = KElbowVisualizer(KMeans(), k=10)
Elbow_M.fit(PCA_ds)
Elbow_M.show()

#senza pca
print('Elbow Method to determine the number of clusters to be formed:')
Elbow_M = KElbowVisualizer(KMeans(), k=10)
Elbow_M.fit(data)
Elbow_M.show()

from sklearn.cluster import KMeans

ebw = []

for i in range(1,10):
    km = KMeans(n_clusters=i)
    km.fit(PCA_ds)
    ebw.append(km.inertia_)

plt.plot(range(1,10),ebw,'g',marker='o')
plt.show()

kmeans = KMeans(n_clusters=4,n_init=10,max_iter=300,random_state=0,tol=1e-04)
kmeans.fit(PCA_ds)
colormap = plt.get_cmap("tab10") 
cluster_colors = colormap(kmeans.labels_)
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(x,y,z,c=cluster_colors)
plt.show()

kmeans2 = KMeans(n_clusters=3)
kmeans2.fit(PCA_ds)
colormap = plt.get_cmap("tab10") 
cluster_colors = colormap(kmeans2.labels_)

plt.scatter(x,y,z,c=cluster_colors)
plt.show()

kmeans3 = KMeans(n_clusters=3,n_init=10,max_iter=300,random_state=0,tol=1e-04)
kmeans3.fit(data)

newData = data
newData["Class"] = kmeans3.labels_

newData0 = newData.loc[newData.Class== 0]
newData1 = newData.loc[newData.Class== 1]
newData2 = newData.loc[newData.Class== 2]

plt.figure(figsize=(19,10))

# Grafico
plt.scatter(newData0.Income, newData0.Spent, color='red', label='Classe A')
plt.scatter(newData1.Income, newData1.Spent, color='blue', label='Classe B')
plt.scatter(newData2.Income, newData2.Spent, color='green', label='Classe C')

# Etichette e titoli
plt.xlabel('Stipendio')
plt.ylabel('Spesa')
plt.title('Scatter plot dello stipendio vs spesa per Classe')
plt.legend()

from sklearn.metrics import silhouette_samples, silhouette_score

X = data  # Usa il tuo dataset originale senza la colonna dei cluster
labels = kmeans3.labels_

# Calcolare il punteggio della silhouette
silhouette_avg = silhouette_score(X, labels)
print("The average silhouette_score is :", silhouette_avg)

# Calcolare i valori di silhouette per ogni campione
sample_silhouette_values = silhouette_samples(X, labels)

y_lower = 10
fig, ax1 = plt.subplots(1, 1)
fig.set_size_inches(18, 7)

# Etichetta sull'asse y
ax1.set_yticks([])
ax1.set_yticklabels([])
ax1.set_ylabel("Cluster")

# Etichetta sull'asse x
ax1.set_xlabel("Silhouette")

# Punti verticali che indicano l'average silhouette score
ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

# Colori per i 3 cluster
colors = cm.nipy_spectral(labels.astype(float) / 3)

for i in range(3):
    # Aggregare i valori della silhouette per ogni cluster
    ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
    ith_cluster_silhouette_values.sort()

    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    color = cm.nipy_spectral(float(i) / 3)
    ax1.fill_betweenx(np.arange(y_lower, y_upper),
                      0, ith_cluster_silhouette_values,
                      facecolor=color, edgecolor=color, alpha=0.7)

    # Etichetta del cluster
    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

    # Calcolare la nuova y_lower per il prossimo plot
    y_lower = y_upper + 10  # 10 per lo spazio bianco tra i plot

plt.title("Silhouette dei cluster")
plt.show()

# Definizione dei colori per i cluster
cluster_colors = ['red', 'blue', 'green']

# Impostare lo stile di seaborn
sns.set_theme(style="whitegrid")

# Creazione del grafico a barre con mappa di colori
plt.figure(figsize=(15, 5))

# Iterazione attraverso i cluster
for cluster in range(3):
    # Seleziona le righe corrispondenti a questo cluster
    cluster_data = newData[newData['Class'] == cluster]
    # Calcola il numero di righe per questo cluster
    count = cluster_data.shape[0]
    # Traccia una barra per questo cluster con il colore corrispondente
    plt.bar(cluster, count, color=cluster_colors[cluster], label=f'Cluster {cluster}')

# Etichette degli assi
plt.xlabel('Cluster')
plt.ylabel('Numero di elementi')
plt.title('Numero di elementi per cluster')
# Aggiungi la legenda
plt.legend()
# Mostra il grafico
plt.show()

sns.set_theme(style="whitegrid")

# Creazione dei boxplot
plt.figure(figsize=(15, 8))

# Boxplot per la classe 0
plt.subplot(1, 3, 3)
sns.boxplot(x="Class", y="Age", data=newData0, color="red")
plt.title('Classe A')

# Boxplot per la classe 1
plt.subplot(1, 3, 1)
sns.boxplot(x="Class", y="Age", data=newData1, color="blue")
plt.title('Classe B')

# Boxplot per la classe 2
plt.subplot(1, 3, 2)
sns.boxplot(x="Class", y="Age", data=newData2, color="green")
plt.title('Classe C')

# Mostra i boxplot
plt.tight_layout()
plt.show()


sns.set(style="whitegrid")

# Creazione dei countplot
plt.figure(figsize=(15, 8))

# Countplot per la classe 0
plt.subplot(1, 3, 3)
sns.countplot(x="Marital_Status", data=newData0, order=newData0["Marital_Status"].value_counts().index)
plt.title('Classe A')
plt.xlabel('Stato matrimoniale')
plt.ylabel('Conteggio')

# Countplot per la classe 1
plt.subplot(1, 3, 1)
sns.countplot(x="Marital_Status", data=newData1, order=newData1["Marital_Status"].value_counts().index)
plt.title('Classe B')
plt.xlabel('Stato matrimoniale')
plt.ylabel('Conteggio')

# Countplot per la classe 2
plt.subplot(1, 3, 2)
sns.countplot(x="Marital_Status", data=newData2, order=newData2["Marital_Status"].value_counts().index)
plt.title('Classe C')
plt.xlabel('Stato matrimoniale')
plt.ylabel('Conteggio')

# Mostra i countplot
plt.tight_layout()
plt.show()

# Creazione dei boxplot
plt.figure(figsize=(15, 8))

# Boxplot per la classe 0
plt.subplot(1, 3, 3)
sns.boxplot(x="Class", y="Income", data=newData0, color='red')
plt.title('Classe A')

# Boxplot per la classe 1
plt.subplot(1, 3, 1)
sns.boxplot(x="Class", y="Income", data=newData1, color='blue')
plt.title('Classe B')

# Boxplot per la classe 2
plt.subplot(1, 3, 2)
sns.boxplot(x="Class", y="Income", data=newData2, color='green')
plt.title('Classe C')

# Mostra i boxplot
plt.tight_layout()
plt.show()

sns.set_theme(style="whitegrid")

# Creazione dei boxplot
plt.figure(figsize=(15, 8))

# Boxplot per la classe 0
plt.subplot(1, 3, 3)
sns.boxplot(x="Class", y="Spent", data=newData0, color='red')
plt.title('Classe A')

# Boxplot per la classe 1
plt.subplot(1, 3, 1)
sns.boxplot(x="Class", y="Spent", data=newData1, color='blue')
plt.title('Classe B')

# Boxplot per la classe 2
plt.subplot(1, 3, 2)
sns.boxplot(x="Class", y="Spent", data=newData2, color='green')
plt.title('Classe C')

# Mostra i boxplot
plt.tight_layout()
plt.show()

sns.set(style="whitegrid")

# Creazione dei subplot per gli istogrammi
plt.figure(figsize=(15, 5))

# Istogramma per la classe 0
plt.subplot(1, 3, 3)
sns.histplot(newData0["Children"], bins=4, color='red')
plt.title('Classe A')
plt.xlabel('Numero di figli')
plt.ylabel('Frequenza')

# Istogramma per la classe 1
plt.subplot(1, 3, 1)
sns.histplot(newData1["Children"], bins=4, color='blue')
plt.title('Classe B')
plt.xlabel('Numero di figli')
plt.ylabel('Frequenza')

# Istogramma per la classe 2
plt.subplot(1, 3, 2)
sns.histplot(newData2["Children"], bins=4, color='green')
plt.title('Classe C')
plt.xlabel('Numero di figli')
plt.ylabel('Frequenza')

# Mostra gli istogrammi
plt.tight_layout()
plt.show()

sns.set_theme(style="whitegrid")

# Creazione dei boxplot
plt.figure(figsize=(15, 8))

# Boxplot per la classe 0
plt.subplot(1, 3, 3)
sns.boxplot(x="Class", y="Children", data=newData0, color='red')
plt.title('Classe A')

# Boxplot per la classe 1
plt.subplot(1, 3, 1)
sns.boxplot(x="Class", y="Children", data=newData1, color='blue')
plt.title('Classe B')

# Boxplot per la classe 2
plt.subplot(1, 3, 2)
sns.boxplot(x="Class", y="Children", data=newData2, color='green')
plt.title('Classe C')

plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection="3d")
colormap = plt.get_cmap("tab10") 
cluster_colors = colormap(kmeans2.labels_)
ax.scatter(x,y,z,c=cluster_colors)
plt.show()

kmeans.labels_

ac = AgglomerativeClustering(n_clusters=4)
predictions = ac.fit_predict(PCA_ds)
colormap = plt.get_cmap("tab10") 
cluster_colors = colormap(predictions)
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(x,y,z,c=cluster_colors,marker='o')
plt.show()

ac2 = AgglomerativeClustering(n_clusters=5)
ac2.fit(data)

from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

silhouette = silhouette_score(PCA_ds, kmeans.labels_)
db_index = davies_bouldin_score(PCA_ds, kmeans.labels_)
ch_index = calinski_harabasz_score(PCA_ds, kmeans.labels_)

print(f"Silhouette Score: {silhouette:.2f}")
print(f"Davies-Bouldin Index: {db_index:.2f}")
print(f"Calinski-Harabasz Index: {ch_index:.2f}")

silhouette = silhouette_score(data, kmeans3.labels_)
db_index = davies_bouldin_score(data, kmeans3.labels_)
ch_index = calinski_harabasz_score(data, kmeans3.labels_)

print(f"Silhouette Score: {silhouette:.2f}")
print(f"Davies-Bouldin Index: {db_index:.2f}")
print(f"Calinski-Harabasz Index: {ch_index:.2f}")

unique, counts = np.unique(kmeans3.labels_, return_counts=True)
dict(zip(unique, counts))

from sklearn.model_selection import train_test_split

train, test, train_labels, test_labels = train_test_split(data,
                                                          kmeans3.labels_,
                                                          test_size=0.30,
                                                          random_state=42)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay

classifier = DecisionTreeClassifier(random_state=17)
classifier.fit(train, train_labels)
pred_y = classifier.predict(test)
cm = confusion_matrix(test_labels,pred_y)
accuracy_score(test_labels,pred_y)

plt.figure(figsize = (10,7))
sns.heatmap(cm, annot=True,fmt='g',cmap="YlGnBu")
plt.show()

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

cv = KFold(n_splits=10, random_state=1, shuffle=True)
model = LogisticRegression()
scores = cross_val_score(model, data, predictions, scoring='accuracy', cv=cv, n_jobs=-1)
print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

print(scores)

model = DecisionTreeClassifier()
scores = cross_val_score(model, data, predictions, scoring='accuracy', cv=cv, n_jobs=-1)
print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

print(scores)

from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

models = []

models.append(("LogisticRegression",LogisticRegression()))
models.append(("SVC",SVC()))
models.append(("LinearSVC",LinearSVC()))
models.append(("KNeighbors",KNeighborsClassifier()))
models.append(("DecisionTree",DecisionTreeClassifier()))
models.append(("RandomForest",RandomForestClassifier()))
rf2 = RandomForestClassifier(n_estimators=100, criterion='gini',
                                max_depth=10, random_state=0, max_features=None)
models.append(("RandomForest2",rf2))
models.append(("MLPClassifier",MLPClassifier(solver='lbfgs', random_state=0)))

results = []
names = []
for name,model in models:
    result = cross_val_score(model, train, train_labels,  cv=3)
    names.append(name)
    results.append(result)

for i in range(len(names)):
    print(names[i],results[i].mean())

model_name, model = models[4]
model.fit(train,train_labels)
pred_y = model.predict(test)
cm = confusion_matrix(test_labels,pred_y)
accuracy_score(test_labels,pred_y)

for model_name, model in models:
    # Adattare il modello
    model.fit(train, train_labels)
    pred_y = model.predict(test)

    # Calcolare la matrice di confusione e l'accuratezza
    cm = confusion_matrix(test_labels, pred_y)
    accuracy = accuracy_score(test_labels, pred_y)

    # Stampare l'accuratezza
    print(f"Accuracy of {model_name}: {accuracy:.2f}")

    # Stampare il report di classificazione
    print(f"Classification Report for {model_name}:\n")
    print(classification_report(test_labels, pred_y))

    # Visualizzare la matrice di confusione
    #disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    #disp.plot(cmap=plt.cm.Blues)
    #plt.title(f'Confusion Matrix for {model_name}')
    #plt.show()

    plt.figure(figsize = (10,7))
    sns.heatmap(cm, annot=True,fmt='g',cmap="YlGnBu")
    plt.show()


