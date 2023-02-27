import requests

url = "https://modeloai.albertosaenz.com/api"
datos = {
    "sepal_length": 5.0,
    "sepal_width": 3.6,
    "petal_length ": 1.4,
    "petal_width": 0.2,
}  
respuesta = requests.post(url, json=datos)
respuesta_json = respuesta.json()
print("La respuesta del servidor es:")
print(respuesta_json)