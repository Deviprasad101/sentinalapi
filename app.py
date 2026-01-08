from flask import Flask, render_template, send_from_directory
import os
import json

app = Flask(__name__)

# Ensure the tiffs directory exists
TIFF_DIR = os.path.join(app.root_path, "static", "tiffs")
os.makedirs(TIFF_DIR, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/tiffs/<path:filename>")
def serve_tiff(filename):
    return send_from_directory("static/tiffs", filename)

@app.route("/stats")
def stats():
    stats_path = os.path.join(TIFF_DIR, "stats.json")
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            return json.load(f)
    return {}

if __name__ == "__main__":
    app.run(debug=True)
