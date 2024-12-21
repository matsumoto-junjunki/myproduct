from flask import Flask, render_template, request, jsonify
from vector2link import search_program
import csv  # csvモジュールをインポート
import os

app = Flask(__name__)

# annのcsvファイル読み込み
def load_csv_ann():
    ann = []
    csv_file = "ann.csv"   
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"{csv_file} が見つかりません")
    with open("ann.csv", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            ann.append(row)
    return ann

# JUNKのcsvファイル読み込み
def load_csv_JUNK():
    JUNK = []
    JUNK_file = "JUNK.csv"   
    if not os.path.exists(JUNK_file):
        raise FileNotFoundError(f"{JUNK_file} が見つかりません")
    with open("JUNK.csv", encoding="utf-8") as JUNKcsvfile:
        reader = csv.DictReader(JUNKcsvfile)
        for row in reader:
            JUNK.append(row)
    return JUNK

@app.route("/")
def index():
    ann_programs = load_csv_ann()
    JUNK_programs = load_csv_JUNK()
    return render_template("index_link.html", ann_programs=ann_programs, JUNK_programs=JUNK_programs)

@app.route("/search", methods=["POST"])
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
