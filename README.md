# Prueba técnica Willinn - IA
Este proyecto tiene la siguiente arquitectura
```
|- graders
| |- __init__.py
| |- answer_grader.py
| |- hallucination_grader.py
| |- retrieval_grader.py
|- routing
| |- __init__.py
| |- router.py
| - test
| |- answer_test.py
| |- hallucinationGrader.py
| |- ragtest.py
| |- retrievalGrader.py
| |- routing.py
| |- testall.py
|- utils
| |- howto-ipaddress.pdf
| |- howto-sorting.pdf
| |- rag.py
| |- web_search.py
|- vectorstore
| |- document_loader.py
|- willinn_langchain
| |- index.faiss
| |- index.pkl
|- .env
|- app.py
|- main.py
|- init_llama.py
|- README.md
|- requirements.txt
```

El diagrama de flujo del proyecto es el siguiente:
![Diagrama de flujo](https://github.com/user-attachments/assets/366a7876-a111-4a53-b78b-2f8797f529a7)

Al ingresar la pregunta, tenemos un **router** que define si el prompt tiene que ver con los documentos que ya hay en la base de datos vectorial.
Si no lo hay, busca en internet y responde.
Si lo hay, devuelve todos los documentos y los clasifica en aquellos que tienen que ver con la pregunta y aquellos que no.
Si no hay documentos que puedan responder esa pregunta, busca en internet y responde.
Si hay documentos que pueden responder, utiliza langchain para responderla.
Luego de responder, se pregunta si está alucinando, si la respuesta es sí, busca en internet es responde.
Si no está alucinando, se pregunta si está respondiendo el prompt que se hizo en un principio.
Si no lo responde, busca en internet y responde.
Si la respuesta está bien, finalmente contesta.

---

### Para correr el proyecto es necesario hacer:
```pip install -r requirements.txt```
Para instalar los requerimentos del proyecto.

```ollama pull llama3.2:1b-instruct-fp16```
Para instalar el modelo local y poder correr el proyecto.

Se sugiere utilizar Postman para poder visualizar la interacción con la API.
