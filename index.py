import os
from flask import Flask, jsonify, request
from runTests import index

app = Flask(__name__)

@app.route('/')
def nao_entre_em_panico():
    # if request.headers.get('Authorization') == '42':
    #     return jsonify({"42": "a resposta para a vida, o universo e tudo mais"})
    return jsonify({"result": index(request.args['original'], request.args['marcada'])})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)