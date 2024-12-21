from flask import Flask, render_template, request, jsonify
from vector2 import search_program
import csv  # csvモジュールをインポート
import os

app = Flask(__name__)


@app.route("/copy")
def index():
    return render_template("index2.html")

@app.route("/search_copy", methods=["POST"])
def search():
    # JSON形式でリクエストを取得
    data = request.get_json()
    query = data.get("query", "").strip() if data else ""

    if not query:
        return jsonify({"error": "検索キーワードを入力してください。"}), 400

    result = search_program(query)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
