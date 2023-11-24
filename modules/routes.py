import os
import pandas as pd

from modules import app
from sklearn.svm import SVC
from flask import render_template, request
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


path = os.path.abspath(os.path.dirname(__file__))
csv_url = f'{path}/data/estudent.csv'
dataset = pd.read_csv(csv_url, sep=';', nrows=5000)


@app.route('/', methods=['GET', 'POST'])
def homepage():
    apresentar_campos = False
    texto_paremtro = ""
    tituloClassificador = ""
    classificador = ""

    if request.method == 'POST':
        classificador = request.form.get('classificador')

        apresentar_campos = classificador in ["KNN", "MLP", "DT", "SVM", "RF"]

        if apresentar_campos:
            if classificador == "KNN":
                tituloClassificador = "K-Nearest Neighbors"
                texto_paremtro = "Selecione um valor para: n_neighbors"
            elif classificador == "DT":
                tituloClassificador = "Decision Tree"
                texto_paremtro = "Selecione um valor para: max_depth"
            elif classificador == "RF":
                tituloClassificador = "Random Forest"
                texto_paremtro = "Selecione um valor para: n_estimators"
            elif classificador == "MLP":
                tituloClassificador = "Multilayer Perceptron"
                texto_paremtro = "Selecione um valor para: hidden_layer_sizes"
            elif classificador == "SVM":
                tituloClassificador = "Support Vector Machine"
                texto_paremtro = "Selecione um valor para: C"

    return render_template(
        'home.html',
        apresentar_campos=apresentar_campos,
        texto_paremtro=texto_paremtro,
        classificador=classificador,
        tituloClassificador=tituloClassificador
    )

@app.route('/resultado', methods=['POST'])
def resultadoPage():
    classificador = request.form.get('classificador')
    parametro = int(request.form.get('parametro'))
    lista_colunas_removidas = ['school','Medu', 'Fedu', 'famsize', 'Mjob', 'Fjob', 'reason', 'traveltime', 'schoolsup', 'paid', 'nursery', 'freetime', 'absences', 'G1', 'G2', 'G3'
]
    dataset = dataset.drop(columns = lista_colunas_removidas)
    dataset2 = pd.DataFrame.copy(dataset)
    dataset2['sex'].unique()
    dataset2['sex'].replace({'F':0, 'M': 1}, inplace=True)
    dataset2['address'].unique()
    dataset2['address'].replace({'U':0, 'R': 1}, inplace=True)
    dataset2['Pstatus'].unique()
    dataset2['Pstatus'].replace({'A':0, 'T': 1}, inplace=True)
    dataset2['guardian'].unique()
    dataset2['guardian'].replace({'mother':0, 'father': 1, 'other':2}, inplace=True)
    dataset2['famsup'].unique()
    dataset2['famsup'].replace({'no':0, 'yes': 1}, inplace=True)
    dataset2['activities'].unique()
    dataset2['activities'].replace({'no':0, 'yes': 1}, inplace=True)
    dataset2['higher'].unique()
    dataset2['higher'].replace({'yes':0, 'no': 1}, inplace=True)
    dataset2['internet'].unique()
    dataset2['internet'].replace({'no':0, 'yes': 1}, inplace=True)
    dataset2['romantic'].unique()
    # Separando as features e o alvo
    X = dataset[['IDADE', 'NOTA_CN', 'NOTA_CH', 'NOTA_LC', 'NOTA_MT', 'NOTA_COMP1', 'NOTA_COMP2', 'NOTA_COMP3', 'NOTA_COMP4', 'NOTA_COMP5']]
    y = (dataset['NOTA_REDACAO'] > dataset['NOTA_REDACAO'].median()).astype(int)

    # Feita a separação entre os dados treinados e testados
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if classificador == "KNN":
        tituloClassificador = "K-Nearest Neighbor"
        model = KNeighborsClassifier(n_neighbors=parametro)

    elif classificador == "DT":
        tituloClassificador = "Decision Tree"
        model = DecisionTreeClassifier(max_depth=parametro)

    elif classificador == "RF":
        tituloClassificador = "Random Forest"
        model = RandomForestClassifier(n_estimators=parametro)

    elif classificador == "MLP":
        tituloClassificador = "Multilayer Perceptron"
        model = MLPClassifier(hidden_layer_sizes=(parametro))

    elif classificador == "SVM":
        tituloClassificador = "Support Vector Machine"
        model = SVC(C=parametro, kernel='linear')

    # Treinando a classificação escolhida
    model.fit(X_train, y_train)

    # Tentando realizar previsões
    y_pred = model.predict(X_test)

    #Aqui gera a matrix de confusão
    conf_matrix = confusion_matrix(y_test, y_pred)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return render_template("resultado.html", accuracy=accuracy, precision=precision, recall=recall, f1=f1, tituloClassificador=tituloClassificador, conf_matrix=conf_matrix)


    
