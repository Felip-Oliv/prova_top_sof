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
csv_url = f'{path}/data/heart_tratado.csv'
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

# ...

# ...

@app.route('/resultado', methods=['POST'])
def resultadoPage():
    classificador = request.form.get('classificador')
    parametro = int(request.form.get('parametro'))

    # Separando as features e o alvo
    X = dataset[['Age', 'Cholesterol', 'HeartDisease']].median().astype(int)
    y = (dataset['MaxHR'] > dataset['Oldpeak'].median()).astype(int)

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

    # Aqui gera a matriz de confusão
    conf_matrix = confusion_matrix(y_test, model.predict(X_test))

    accuracy = accuracy_score(y_test, model.predict(X_test))
    precision = precision_score(y_test, model.predict(X_test))
    recall = recall_score(y_test, model.predict(X_test))
    f1 = f1_score(y_test, model.predict(X_test))

    # Convertendo arrays numpy para listas
    X_test_list = X_test.values.tolist()
    y_test_list = y_test.tolist()
    y_pred_list = model.predict(X_test).tolist()

    return render_template("resultado.html", accuracy=accuracy, precision=precision, recall=recall, f1=f1, tituloClassificador=tituloClassificador, conf_matrix=conf_matrix, X_test=X_test_list, y_test=y_test_list, y_pred=y_pred_list)
