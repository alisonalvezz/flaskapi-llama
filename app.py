from flask import Flask, request, jsonify
from main import main

app = Flask(__name__)

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    query_text = data.get("query", "")
    
    if not query_text:
        return jsonify({"error": "No query provided."}), 400

    response = main(query_text)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000)  