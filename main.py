import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA


uploaded_file = st.file_uploader("Choisissez un fichier CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("Invistico_Airline.csv")

st.title("Analyse des Données de Satisfaction Client")

st.write("Aperçu des données :")
st.dataframe(df.head())

st.write("Informations sur les données :")
st.text(df.info())

st.write("Valeurs manquantes :")
st.bar_chart(df.isnull().sum())

st.write("Visualisation des données numériques :")
for col in df.select_dtypes('number').columns:
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    sns.histplot(df[col], ax=axes[0])
    sns.boxplot(df[col], ax=axes[1], showmeans=True)
    st.pyplot(fig)

def impute_outliers(data, colname): 
    q1 = np.percentile(data[colname], 25)
    q3 = np.percentile(data[colname], 75) 

    lower_bound = q1 - 1.5 * (q3 - q1)
    upper_bound = q3 + 1.5 * (q3 - q1)

    data.loc[(data[colname] <= lower_bound), colname] = lower_bound
    data.loc[(data[colname] >= upper_bound), colname] = upper_bound

for colname in df.select_dtypes('number').columns:
    impute_outliers(df, colname)

class DataPreprocessor:
    def __init__(self, df, target_column, exclude_columns=None, test_size=0.2, random_state=0):
        self.df = df
        self.target_column = target_column
        self.exclude_columns = exclude_columns if exclude_columns is not None else []
        self.test_size = test_size
        self.random_state = random_state

    def split_data(self):
        y = self.df[self.target_column]
        X = self.df.drop([self.target_column] + self.exclude_columns, axis='columns')
        
        X_train, X_test, y_train, y_test = train_test_split(
                                 X, 
                                 y, 
                                 test_size=self.test_size, 
                                 random_state=self.random_state
                            )
        return X_train, X_test, y_train, y_test

    def create_pipeline(self, X_train):
        num_cols = X_train.select_dtypes(include=['number']).columns
        cat_cols = X_train.select_dtypes(include='object').columns
        num_pipeline = make_pipeline(
                    SimpleImputer(strategy='median'),
                    StandardScaler(),
                )
        cat_pipeline = make_pipeline(
                    SimpleImputer(strategy='most_frequent'),
                    OneHotEncoder(handle_unknown='ignore', drop='first')
                )
        full_pipeline = make_column_transformer(
                (num_pipeline, num_cols),
                (cat_pipeline, cat_cols)
            )
        return full_pipeline


target_column = st.selectbox("Choisissez la colonne cible :", df.columns)


preprocessing = DataPreprocessor(df, target_column, random_state=10)
X_train, X_test, y_train, y_test = preprocessing.split_data()
pipeline = preprocessing.create_pipeline(X_train)

X_train_transform = pipeline.fit_transform(X_train)
X_test_transform = pipeline.transform(X_test)

n_components = st.number_input("Choisissez le nombre de composants pour l'ACP :", value=3)

pca = PCA(n_components=n_components)
X_train_reduce = pca.fit_transform(X_train_transform)
X_test_reduce = pca.transform(X_test_transform)

display_option = st.radio("Choisissez le type d'affichage :", ('2D', '3D'))

if display_option == '2D': 
    st.write("Visualisation 2D des données projetées :")
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)

    plt.figure(figsize=(20, 15))
    scatter = plt.scatter(X_train_reduce[:, 0], X_train_reduce[:, 1], 
                        c=y_train_encoded, cmap='coolwarm', s=40)
    plt.xlabel(f'Première composante principale ({pca.explained_variance_ratio_[0]:.2%} variance expliquée)')
    plt.ylabel(f'Deuxième composante principale ({pca.explained_variance_ratio_[1]:.2%} variance expliquée)')
    plt.title('Projection ACP avec visualisation des classes')
    st.pyplot(plt)
elif display_option == '3D':
    df_pca = pd.DataFrame(X_train_reduce, columns=['PC1', 'PC2', 'PC3'])
    df_pca[target_column] = y_train

    def plot_3d_individuals(df_pca, pca):
        fig = plt.figure(figsize=(30, 30))
        le = LabelEncoder()
        ax = fig.add_subplot(111, projection='3d')
        satisfaction_encoded = le.fit_transform(df_pca[target_column])
        scatter = ax.scatter(df_pca['PC1'], df_pca['PC2'], df_pca['PC3'],
                            c=satisfaction_encoded, cmap='coolwarm', s=80, alpha=0.6)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
        ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]:.2%} variance)")
        ax.set_title('Visualisation 3D des composantes principales')
        st.pyplot(fig)

    plot_3d_individuals(df_pca, pca)


model_choice = st.sidebar.selectbox("Choisissez le modèle :", ['KNN', 'Gaussian Naive Bayes'])

if model_choice == 'KNN':
    k_neighbors = st.sidebar.slider("Choisissez le nombre de voisins (k) :", min_value=1, max_value=20, value=5)
    model_knn = KNeighborsClassifier(n_neighbors=k_neighbors)
    model_knn.fit(X_train_reduce, y_train)
    y_pred_knn = model_knn.predict(X_test_reduce)
    rapport_KNN = classification_report(y_test, y_pred_knn, output_dict=True)
    rapport_KNN_df = pd.DataFrame(rapport_KNN).T
 
    st.write("Rapport KNN :")
    st.dataframe(rapport_KNN_df)


    if st.button("Afficher les meilleurs hyper paramètres"):
        param_grid = {'n_neighbors': range(1, 20), 'metric': ['minkowski', 'euclidean', 'manhattan'], 'weights': ['uniform', 'distance']}
        grid_search = GridSearchCV(model_knn, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train_reduce, y_train)
        st.write("Meilleurs hyper-paramètres :", grid_search.best_params_)
        st.write(f"Meilleure performance : {grid_search.best_score_:.2f}")

        model_knn_opt = KNeighborsClassifier(n_neighbors=grid_search.best_params_['n_neighbors'], 
                                            metric=grid_search.best_params_['metric'], 
                                            weights=grid_search.best_params_['weights'])
        model_knn_opt.fit(X_train_reduce, y_train)
        y_pred_knn_opt = model_knn_opt.predict(X_test_reduce)
        rapport_KNN_opt = classification_report(y_test, y_pred_knn_opt, output_dict=True)
        rapport_KNN_opt_df = pd.DataFrame(rapport_KNN_opt).T
 
        st.write("Rapport KNN Optimisé :")
        st.dataframe(rapport_KNN_df)

        st.write("Tableau recapitulatif du model Knn et sa version Optimisé")
        performance_summary = pd.DataFrame({
            'Modèle': ['KNN', 'KNN Optimisé'],
            'Précision': [classification_report(y_test, y_pred_knn, output_dict=True)['accuracy'], 
                        classification_report(y_test, y_pred_knn_opt, output_dict=True)['accuracy']],
        })
        st.write(performance_summary)

elif model_choice == 'Gaussian Naive Bayes':
    model_NB = GaussianNB()
    model_NB.fit(X_train_reduce, y_train)
    y_pred_NB = model_NB.predict(X_test_reduce)
    rapport_NB = classification_report(y_test, y_pred_NB, output_dict=True)
    rapport_NB_df = pd.DataFrame(rapport_NB).T

    st.write("Rapport Naive Bayes :")
    st.dataframe(rapport_NB_df)
