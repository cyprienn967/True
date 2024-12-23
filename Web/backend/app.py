from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Simulate a database for demo purposes
user_data = {}

@app.route("/api/store_key", methods=["POST"])
def store_key():
    data = request.json
    username = data.get("username")
    api_key = data.get("api_key")
    if not username or not api_key:
        return jsonify({"error": "Username and API key are required"}), 400
    user_data[username] = api_key
    return jsonify({"message": "API key stored successfully"})

@app.route("/api/get_key", methods=["GET"])
def get_key():
    username = request.args.get("username")
    if username in user_data:
        return jsonify({"api_key": user_data[username]})
    return jsonify({"api_key": ""})

if __name__ == "__main__":
    app.run(debug=True)
