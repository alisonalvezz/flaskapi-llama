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
---

## El diagrama de flujo del proyecto es el siguiente:

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

El proyecto se inicia ejecutando ```app.py```

Se sugiere utilizar Postman para poder visualizar la interacción con la API.
Es necesario que la consulta se realice en formato JSON, ej:
```
{
    "query": "¿Qué es Willinn?"
}
```
La respuesta será algo así:
```
{
    "response": "Willinn es una empresa de tecnología que nació como parte de Zonamerica en Uruguay. Se especializa en servicios de software, ciberseguridad y soluciones de tecnología para empresas en Latinoamérica y a nivel mundial. Con 30 años de experiencia, Willinn se enfoca en llevar a las empresas al siguiente nivel a través de soluciones efectivas que van desde el negocio a la tecnología, incluyendo servicios en la nube, ciberseguridad y operaciones de TI. Willinn cuenta con 350 clientes y proyecta un crecimiento continuo en América."
}
```

---

### Mejoras del proyecto:
Las mejoras que pueden hacerse en el proyecto son:
- Mejorar prompts dados a la IA para mayor exactitud en las respuestas. Por ejemplo: Cuando se pregunta si la pregunta responde a la respuesta, o cuando se pregunta si la respuesta está alucinada. Mejorar esto ayudaría significativamente la precisión de clasificación de las preguntas (si está alucinada o no, si tiene que ver con los vectorstores, etcétera).
