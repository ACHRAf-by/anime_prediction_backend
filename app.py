# Dependencies
from flask import Flask, request
import pandas as pd
import jsonschema
from jsonschema import validate

# API definition
app = Flask(__name__)


# Define JSON schema
schema = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "gender": {"type": "array", "items": {"type": "string"}},
        "description": {"type": "string"},
        "type": {"type": "integer", "enum": [0, 1]},
        "producer": {"type": "string"},
        "studio": {"type": "string"}
    },
}

@app.route('/predict', methods=['POST'])
def predict():
    json_data = request.get_json()
    try:
        validate(instance=json_data, schema=schema)
    except jsonschema.exceptions.ValidationError as err:
        return {'message': err.message}, 400
    # process the valid json_data here
    return json_data
    
if __name__ == '__main__':
    app.run(debug=True)