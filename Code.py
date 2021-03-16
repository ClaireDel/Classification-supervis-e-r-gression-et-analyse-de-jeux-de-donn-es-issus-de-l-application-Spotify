import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import plot_confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import svm
from sklearn.linear_model import Ridge
from sklearn.linear_model import BayesianRidge



#---------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------#



#                                         Exercice 1



# Extraction des jeux de données de l'exercice 1
Train = pd.read_csv("Spotify_train_dataset.csv")
Test = pd.read_csv("Spotify_test_dataset.csv")




def examen(Tableau):
    print ("Taille =",Tableau.shape)
    print (Tableau.info())
    print (Tableau.isnull().sum())
    print (Tableau.corr())


# On enlève les colonnes inutiles
def preprocessing() :
    Train.drop(['type','id','uri','track_href','analysis_url','song_name'], axis='columns', inplace=True)
    return Train

Train = preprocessing()






# Définition préliminaire des jeux de données
y = Train.iloc[:,13]
x = Train.drop('genre', axis=1) 




# Centrage et réduction des données de x
sc = StandardScaler()
x = sc.fit_transform(x)


# Découpe en jeux de test et de train
# Tous les algorithmes utiliseront cette même découpe, pour assurer une cohérence des résultats
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=0)


#                                    ALGORITHMES
#---------------------------------------------------------------------------------------------------------------# 


def arbre() :
    print('CLASSIFICATION PAR ARBRE DECISIONNEL')
    clf = DecisionTreeClassifier(random_state=0, criterion='entropy', max_depth=11)
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test, check_input=True)
    print('Précision : ' + str(100*(f1_score(y_test, y_pred, average='micro'))) + " %")
    print("Profondeur de l'arbre : " + str(clf.get_depth()))
        
    plot_confusion_matrix(clf, x_test, y_test, cmap=plt.cm.Blues) 
    plt.title('Matrice de confusion')
    plt.show()  
  
  
#---------------------------------------------------------------------------------------------------------------# 


def foret() :
    print('CLASSIFICATION PAR FORET ALEATOIRE')
    clf = RandomForestClassifier(random_state=0)
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    print('Précision : ' + str(100*(f1_score(y_test, y_pred, average='micro'))) + " %")
    
    plot_confusion_matrix(clf, x_test, y_test, cmap=plt.cm.Blues) 
    plt.title('Matrice de confusion')
    plt.show()


#---------------------------------------------------------------------------------------------------------------# 


def LDA() :  
    print('CLASSIFICATION PAR ANALYSE DISCRIMINANTE LINEAIRE')
    clf = LinearDiscriminantAnalysis()
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    print('Précision : ' + str(100*(f1_score(y_test, y_pred, average='micro'))) + " %")
    
    plot_confusion_matrix(clf, x_test, y_test, cmap=plt.cm.Blues) 
    plt.title('Matrice de confusion')
    plt.show()

  
#---------------------------------------------------------------------------------------------------------------# 


def QDA() :
    print('CLASSIFICATION PAR ANALYSE DISCRIMINANTE QUADRATIQUE')
    clf = QuadraticDiscriminantAnalysis()
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    print('Précision : ' + str(100*(f1_score(y_test, y_pred, average='micro'))) + " %")
    
    plot_confusion_matrix(clf, x_test, y_test, cmap=plt.cm.Blues) 
    plt.title('Matrice de confusion')
    plt.show()

  
#---------------------------------------------------------------------------------------------------------------# 


def class_bayes() :   
    print('CLASSIFICATION NAIVE BAYESIENNE')
    clf = GaussianNB()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print('Précision : ' + str(100*(f1_score(y_test, y_pred, average='micro'))) + " %")
    
    plot_confusion_matrix(clf, x_test, y_test, cmap=plt.cm.Blues) 
    plt.title('Matrice de confusion')
    plt.show()

  
#---------------------------------------------------------------------------------------------------------------# 


def reglog() :   
    print('CLASSIFICATION PAR REGRESSION LOGISTIQUE')
    clf = LogisticRegression(random_state=0, max_iter=1000)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print('Précision : ' + str(100*(f1_score(y_test, y_pred, average='micro'))) + " %")
    
    plot_confusion_matrix(clf, x_test, y_test, cmap=plt.cm.Blues) 
    plt.title('Matrice de confusion')
    plt.show()


#---------------------------------------------------------------------------------------------------------------# 


def perceptron() :
    print('CLASSIFICATION PAR PERCEPTRON')
    clf = Perceptron(random_state=0)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print('Précision : ' + str(100*(f1_score(y_test, y_pred, average='micro'))) + " %")
    
    plot_confusion_matrix(clf, x_test, y_test, cmap=plt.cm.Blues) 
    plt.title('Matrice de confusion')
    plt.show()


#---------------------------------------------------------------------------------------------------------------# 


def class_MLP() :
    print('CLASSIFICATION PAR RESEAU NEURONAL (MULTILAYER PERCEPTRON)')
    clf = MLPClassifier(max_iter=500, random_state=0, hidden_layer_sizes=(150, 150, 100, 50, 15))
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print('Results on the test set:')
    print(classification_report(y_test, y_pred))
    print('Précision : ' + str(100*(f1_score(y_test, y_pred, average='micro'))) + " %")
    
    plot_confusion_matrix(clf, x_test, y_test, cmap=plt.cm.Blues) 
    plt.title('Matrice de confusion')
    plt.show()


#---------------------------------------------------------------------------------------------------------------# 


def svc() :
    print('CLASSIFICATION PAR SVC')
    clf = SVC(random_state=0)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print('Précision : ' + str(100*(f1_score(y_test, y_pred, average='micro'))) + " %")
    
    plot_confusion_matrix(clf, x_test, y_test, cmap=plt.cm.Blues) 
    plt.title('Matrice de confusion')
    plt.show()


#---------------------------------------------------------------------------------------------------------------# 


def bootstrap() :
    print('CLASSIFICATION PAR BOOTSTRAP AGGREGATING')
    clf = BaggingClassifier(random_state=0)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print('Précision : ' + str(100*(f1_score(y_test, y_pred, average='micro'))) + " %")
    
    plot_confusion_matrix(clf, x_test, y_test, cmap=plt.cm.Blues) 
    plt.title('Matrice de confusion')
    plt.show()


#---------------------------------------------------------------------------------------------------------------# 


def extra_trees() :
    print('CLASSIFICATION PAR EXTRA TREES')
    clf = ExtraTreesClassifier(n_estimators=1000, random_state=0)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print('Précision : ' + str(100*(f1_score(y_test, y_pred, average='micro'))) + " %")
    
    plot_confusion_matrix(clf, x_test, y_test, cmap=plt.cm.Blues) 
    plt.title('Matrice de confusion')
    plt.show()


#---------------------------------------------------------------------------------------------------------------# 


def adaboost() :
    print('CLASSIFICATION PAR ADABOOST')
    clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print('Précision : ' + str(100*(f1_score(y_test, y_pred, average='micro'))) + " %")
    
    plot_confusion_matrix(clf, x_test, y_test, cmap=plt.cm.Blues) 
    plt.title('Matrice de confusion')
    plt.show()


#---------------------------------------------------------------------------------------------------------------# 


def gradient() :
    print('CLASSIFICATION PAR GRADIENT BOOSTING')
    clf = GradientBoostingClassifier(n_estimators=100, random_state=0)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print('Précision : ' + str(100*(f1_score(y_test, y_pred, average='micro'))) + " %")
    
    plot_confusion_matrix(clf, x_test, y_test, cmap=plt.cm.Blues) 
    plt.title('Matrice de confusion')
    plt.show()
    
    
    

#---------------------------------------------------------------------------------------------------------------# 


def k_means()  :
    print('CLASSIFICATION PAR K PLUS PROCHES VOISINS')
    clf = KNeighborsClassifier(n_neighbors=10)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print('Précision : ' + str(100*(f1_score(y_test, y_pred, average='micro'))) + " %")
    
    plot_confusion_matrix(clf, x_test, y_test, cmap=plt.cm.Blues) 
    plt.title('Matrice de confusion')
    plt.show()







#---------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------#



#                                         Exercice 2


# Extraction du jeu de données
df = pd.read_csv('Spotify_exo2.csv')



# Répartition des valeurs de popularité
def hist():
    df.loc[:,'popularity'].plot.hist()
    plt.title('Répartition des scores de popularité du jeu de données')
    plt.xlabel('Score de popularité')
    plt.ylabel('Fréquence')
    plt.show()
    plt.figure(figsize=(20, 20))




# Définition préliminaire des jeux de données x2 et y2

y2 = df.iloc[:, 12]
y2 = pd.Series(y2).to_numpy()
y2 = y2.reshape(2973,-1)

# On enlève les colonnes inutiles
x2 = df.drop(['popularity', 'genres'], axis=1)

# On centre et réduit les données pour faciliter la régression
x2 = StandardScaler().fit_transform(x2)

# Création des jeux d'entrainement et de test
x2_train,x2_test,y2_train,y2_test=train_test_split(x2,y2,test_size=0.2, random_state=0)
y2_train = np.ravel(y2_train)
y2_test = np.ravel(y2_test)




#                                    ALGORITHMES
#---------------------------------------------------------------------------------------------------------------# 


def mlp_reg():
    print('')
    print('REGRESSION PAR RESEAU NEURONAL (MULTILAYER PERCEPTRON)')
    reg = MLPRegressor(max_iter=300, random_state=0, activation='relu')
    reg.fit(x2_train, y2_train)
    y_pred = reg.predict(x2_test)
    print('RMSE :', np.sqrt(mean_squared_error(y2_test, y_pred)))
    print('R2 :', r2_score(y2_test, y_pred))


#---------------------------------------------------------------------------------------------------------------# 


def SVM():
    print('')
    print('REGRESSION VECTORIELLE (SVM)')
    reg = svm.SVR()
    reg.fit(x2_train, y2_train)
    y_pred = reg.predict(x2_test)
    print('RMSE :', np.sqrt(mean_squared_error(y2_test, y_pred)))
    print('R2 :', r2_score(y2_test, y_pred))


#---------------------------------------------------------------------------------------------------------------# 


def bayes():
    print('')
    print('REGRESSION NAIVE DE BAYES')
    reg = BayesianRidge()
    reg.fit(x2_train, y2_train)
    y_pred = reg.predict(x2_test)
    print('RMSE :', np.sqrt(mean_squared_error(y2_test, y_pred)))
    print('R2 :', r2_score(y2_test, y_pred))
    
    
#---------------------------------------------------------------------------------------------------------------# 


# Fonction préliminaire à la regression par ridge, permettant de déterminer le alpha optimal
def optimal() :
    L=np.arange(1,10**4,1)
    RMSE = []
    for i in range(len(L)):
        reg = Ridge(alpha=L[i],random_state=None,tol=10**-10,solver='sag')
        reg.fit(x2_train, y2_train)
        y_pred = reg.predict(x2_test)
        RMSE.append(np.sqrt(mean_squared_error(y2_test, y_pred)))
    
    plt.figure(figsize=(10, 10))
    plt.semilogx(L, RMSE)
    plt.ylabel("regression coefficients")
    plt.xlabel("lambda")
    plt.title("RIDGE RMSE coefficients versus the regularization parameter")
    plt.show()
    index_ridge = [i for i, a in enumerate(RMSE) if a == min(RMSE)]
    return L[index_ridge] 
# La valeur retournée d'alpha est 104, c'est celle-ci que nous utilisons en paramètre de la fonction ridge


def ridge():
    print('')
    print('REGRESSION RIDGE')
    reg = Ridge(alpha=104,random_state=None,tol=10**-10,solver='sag')
    reg.fit(x2_train, y2_train)
    y_pred = reg.predict(x2_test)
    print('RMSE :', np.sqrt(mean_squared_error(y2_test, y_pred)))
    print('R2 :', r2_score(y2_test, y_pred))


#---------------------------------------------------------------------------------------------------------------# 




# Vérification d'overfitting via cross validation
def verif_over(model):
    kf = KFold(n_splits=20)
    list_training_error = []
    list_testing_error = []
    for train_index, test_index in kf.split(x2):
        x2_train, x2_test = x2[train_index], x2[test_index]
        y2_train, y2_test = y2[train_index], y2[test_index]
        y2_train = np.ravel(y2_train)
        y2_test = np.ravel(y2_test)
        model.fit(x2_train, y2_train)
        y_train_data_pred = model.predict(x2_train)
        y_test_data_pred = model.predict(x2_test)
    
    
        fold_training_error = np.sqrt(mean_squared_error(y2_train, y_train_data_pred)) 
        fold_testing_error = np.sqrt(mean_squared_error(y2_test, y_test_data_pred))
        list_training_error.append(fold_training_error)
        list_testing_error.append(fold_testing_error)
    
    plt.subplot(1,2,1)
    plt.plot(range(1, kf.get_n_splits() + 1), np.array(list_training_error).ravel(), c= 'blue')
    plt.xlabel('number of fold')
    plt.ylabel('training error')
    plt.title('Training error across folds')
    plt.tight_layout()
    # plt.subplot(1,2,2)
    plt.plot(range(1, kf.get_n_splits() + 1), np.array(list_testing_error).ravel(), c='red')
    plt.xlabel('number of fold')
    plt.ylabel('testing error')
    plt.title('Testing error across folds')
    plt.tight_layout()
    plt.show()
    
    



# Graphes de prédiction en fonction de la complexité    
def complexity(n_max):
    list_training_error = []
    list_testing_error = []
    for i in range(1,n_max):
        model= MLPRegressor(max_iter=i, random_state=0, activation='relu')
        model.fit(x2_train, y2_train)
        y_train_data_pred = model.predict(x2_train)
        y_test_data_pred = model.predict(x2_test)
           
    
        fold_training_error =np.sqrt(mean_squared_error(y2_train, y_train_data_pred))
        fold_testing_error =np.sqrt(mean_squared_error(y2_test, y_test_data_pred))
        list_training_error.append(fold_training_error)
        list_testing_error.append(fold_testing_error)
    
    plt.subplot(1,2,1)
    plt.plot(range(1, n_max), np.array(list_training_error).ravel(), c = 'blue', linestyle = 'dashed')
    plt.xlabel('Model Complexity')
    plt.ylabel('RMSE')
    plt.title('Training Sample')
    plt.tight_layout()
    # plt.subplot(1,2,2)
    plt.plot(range(1, n_max), np.array(list_testing_error).ravel(), c = 'red', linestyle = 'dashed')
    plt.xlabel('Model Complexity')
    plt.ylabel('RMSE')
    plt.title('Testing Sample')
    plt.tight_layout()
    plt.show()
    




# Comparaison des valeurs réelles et des valeurs prédites du fichier (en nuage de points ou pas)
def trace_comparaison(model):
    model.fit(x2_train, y2_train)
    y_pred = model.predict(x2)
    plt.title("Superposition des valeurs réelles et prédites de popularité")
    plt.plot(np.arange(0, len(y2),1),y2, c='red', label="Valeurs réelles")
    plt.plot(np.arange(0, len(y_pred),1), y_pred, c='black', label="Valeurs prédites")
    plt.legend()
    







