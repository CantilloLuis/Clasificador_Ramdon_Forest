#Librerias utilizadas
from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import matplotlib
import openai
import seaborn as sns
from sklearn import tree

matplotlib.use('agg')  # Evitar la inicialización de Tkinter

# Carga el dataset
df = pd.read_excel(r"C:\Users\canti\Downloads\ataques_corazon.xlsx")

def sano_enfermo(fila):
  result = 'enfermo'
  if int(fila['Heart Attack Risk']) == 0:
    result = 'no enfermo'
  return result

df['estado'] = df.apply(sano_enfermo,axis=1)

resultado = dict(zip(df['Heart Attack Risk'].unique(),df['estado'].unique()))
print(resultado)

# Metodo para generar graficos de barras
def generate_bar_chart():
  estado_counts = df['estado'].value_counts()
  # Genera una paleta de colores con la cantidad de colores igual a la cantidad de barras
  colors = sns.color_palette('husl', n_colors=len(estado_counts))
  plt.bar(estado_counts.index, estado_counts.values,color=colors)
  plt.xlabel('Estado')
  plt.ylabel('Cantidad de personas')
  plt.title('Distribución de estado')
  plt.savefig('static/bar_chart.png')  # Guardar el gráfico como imagen
  plt.close()
  
# Después de cargar tus datos y seleccionar las características relevantes, puedes calcular la matriz de correlación
correlation_matrix = df[['Age', 'Cholesterol', 'Heart_Rate', 'Diabetes', 'Smoking', 'Obesity', 'Alcohol_Consumption', 'Previous_Heart_Problems', 'Medication_Use', 'Stress_Level', 'Triglycerides', 'Physical_Activity_Days_Per_Week', 'Sleep_Hours_Per_Day']].corr()

# Metodo para generar graficos de mapa de calor
def generate_bar_heatmap():
  plt.figure(figsize=(10, 8))
  sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 10})
  plt.title('Matriz de correlación')
  plt.savefig('static/heatmap.png')  # Guardar el mapa de calor como imagen
  plt.close()

# Metodo para generar graficos de mapa de dispersion
def generate_bar_dispersion():
  sampled_data = df.sample(n=1000)  # Tomar una muestra aleatoria de 1000 puntos
  plt.scatter(sampled_data['Age'], sampled_data['Heart_Rate'],s=5,alpha=0.5)
  plt.xlabel('Age')
  plt.ylabel('Cholesterol')
  plt.title('Gráfico de Dispersión de Age vs Cholesterol')
  plt.savefig('static/dispersion.png')  # Guardar el mapa de calor como imagen
  plt.close()

#Dividimos el conjunto de datos en: Conjunto de datos independiente (x)
#Conjunto de datos dependiente (y)

X = df[['Age','Cholesterol','Heart_Rate','Diabetes','Smoking','Obesity','Alcohol_Consumption','Previous_Heart_Problems','Medication_Use','Stress_Level','Triglycerides','Physical_Activity_Days_Per_Week','Sleep_Hours_Per_Day']]

y = df['Heart Attack Risk']

#Preparamos el data set para entrenamiento y prueba.
#Dividimos el 80% para entrenamiento y 20% para pruebas

X_train, X_test, y_train,y_test = train_test_split(X,y,random_state = 0,test_size = 0.20)

# Entrenamiento del modelo
forest = RandomForestClassifier(n_estimators = 10,criterion='gini')
forest.fit(X_train,y_train)

arbol_individual = forest.estimators_[0]  # Obtener el primer árbol del Random Forest
nombres_columnas = ['Age', 'Cholesterol', 'Heart_Rate', 'Diabetes', 'Smoking', 'Obesity', 'Alcohol_Consumption', 'Previous_Heart_Problems', 'Medication_Use', 'Stress_Level', 'Triglycerides', 'Physical_Activity_Days_Per_Week', 'Sleep_Hours_Per_Day']

def generar_arbol_decision():
  plt.figure(figsize=(12, 8))
  tree.plot_tree(arbol_individual, feature_names=nombres_columnas, filled=True, rounded=True,fontsize=7,max_depth=3)
  plt.savefig('static/arbol_decision.png',dpi=300)  # Guardar el gráfico como imagen
  plt.close()

generar_arbol_decision()

estimacion_prueba = forest.score(X_test,y_test)
estimacion_entrenamiento = forest.score(X_train,y_train)
print(f'La efectividad con la data de prueba con el modelo RandomForest es de {estimacion_prueba} y con la data de entranamiento es de {estimacion_entrenamiento}')


# prediccion = forest.predict([[25,356,75,1,1,0,0,1,0,0,8,180,7,4]])
# print(resultado[prediccion[0]])

#Crear función para diferentes modelos
def models(X_train, y_train):
  #Using GaussianNB
  from sklearn.naive_bayes import GaussianNB
  gauss = GaussianNB()
  gauss.fit(X_train, y_train)

  #Using DecisionTreeClassifier

  from sklearn.tree import DecisionTreeClassifier
  tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
  tree.fit(X_train, y_train)


  #imprimir la precisión del modelo en los datos de entrenamiento.
  print("[1]Gaussian Naive Bayes Precisión:", gauss.score(X_train, y_train))
  print("[2]Decision Tree Classifier Precisión:", tree.score(X_train, y_train))
  print("[3]Random Forest Classifier Precisión:", forest.score(X_train, y_train))
  return gauss, tree, forest

model = models(X_train, y_train)

# Creación de la aplicación Flask
app = Flask(__name__)

app.static_folder = 'static'

# Ruta principal para la página de inicio
@app.route('/')
def index():
  return render_template('index.html')

# Ruta para generar el grafico de barras
@app.route('/generate_chart', methods=['POST'])
def generate_chart():
  generate_bar_chart()
  return jsonify({'chart_generated': True})

# Ruta para generar el grafico de mapa de calor
@app.route('/generate_heatmap', methods=['POST'])
def generate_heatmap():
  generate_bar_heatmap()
  return jsonify({'generate_heatmap': True})

# Ruta para generar el grafico de dispersion
@app.route('/generate_dispersion', methods=['POST'])
def generate_dispersion():
  generate_bar_dispersion()
  return jsonify({'generate_dispersion': True})

# Ruta para la predicción
@app.route('/predict', methods=['POST'])
def predict():
    # Capturamos los datos enviados
    data = request.get_json()
    caracteristicas = data['caracteristicas']

    # Convertir los síntomas en un DataFrame
    caracteristicas_df = pd.DataFrame(caracteristicas, index=[0])

    print(caracteristicas_df)

    # Realizar la predicción
    prediccion = forest.predict(caracteristicas_df)

    # Devolver el resultado
    return jsonify({
      'prediction': resultado[prediccion[0]],
      'estimacion_prueba': estimacion_prueba,
      'estimacion_entrenamiento': estimacion_entrenamiento
    })

# Api Key extraida de la pagina de openai
openai.api_key = "sk-proj-ccwOsumoeKDu9Z2tpjeQT3BlbkFJNYAqpAVAs9fw5BC7P3qG"

#Ruta para realizarle preguntas a la api con el modelo gpt-3.5-turbo y este nos de un respuesta, la cual se envia a la vista y se muestre en el chat.
@app.route('/obtener_recomendaciones', methods=['POST'])
def obtener_recomendaciones():
    prediccion = request.data.decode('utf-8')  # Obtiene el valor como texto directamente

    mensaje_chatgpt = f"La predicción del modelo es: {prediccion}. ¿Qué recomendaciones puedes dar basadas en esto?"

    respuesta_chatgpt = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Eres un experto en dar recomendaciones personalizadas."},
            {"role": "user", "content": mensaje_chatgpt}
        ]
    )

    recomendaciones = respuesta_chatgpt.choices[0].message.content
    return jsonify({'recomendaciones': recomendaciones})

if __name__ == '__main__':
    app.run(debug=True)
