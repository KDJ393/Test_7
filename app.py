# Importación de bibliotecas para crear una aplicación web con Flask
from flask import Flask
from flask import request
import pandas as pd 
import joblib

# Inicializar la aplicación Flask
app = Flask(__name__)

# Definir una ruta y función para predecir basado en un modelo previamente entrenado
@app.route('/predecir', methods=['POST'])
def predecir():
    json_ = request.json
    df = pd.DataFrame(json_, index=[0])
    query = pd.get_dummies(df)

    # Cargar el modelo entrenado previamente
    clasificador = joblib.load('classifier.pkl')
    prediccion = clasificador.predict(query)

    if prediccion[0] == True:
        return "El pasajero pudo haber sobrevivido al titanic"
    else:
        return "El pasajero pudo NO haber sobrevivido al titanic"

# Si este script se ejecuta como el programa principal, se inicia la aplicación Flask
if __name__ == 'main':
    app.run(port=7000, debug=True)
    # app.run(host=os.getenv('IP', '0.0.0.0'), port=int(os.getenv('PORT', 4444))) - Cuando ya se pone en producción hay que remplazar la linea anterior por esta