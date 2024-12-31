from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/')
def home():
  return jsonify({"message": "Welcome to the CoNLI API!"})


@app.route('/status', methods=['GET'])
def status():
  return jsonify({"status": "API is running"})


@app.route('/predict', methods=['POST'])
def predict():
  data = request.get_json()
  # Add your prediction logic here
  prediction = "dummy_prediction"  # Replace with actual prediction logic
  return jsonify({"prediction": prediction})

if __name__ == '__main__':
  app.run(debug=True)