# Dependencies
from flask import Flask, request, jsonify
import pandas as pd
import jsonschema
from jsonschema import validate
import random
from flask_cors import CORS
from flask_swagger_ui import get_swaggerui_blueprint
from prometheus_flask_exporter import PrometheusMetrics


# API definition
app = Flask(__name__)
PrometheusMetrics(app)
CORS(app, origins="*")

SWAGGER_URL = "/api/swagger"
API_URL = "/static/swagger.json"
SWAGGER_BLUEPRINT = get_swaggerui_blueprint(
    SWAGGER_URL, API_URL, config={"app_name": "Anime_Rating_API"}
)

app.register_blueprint(SWAGGER_BLUEPRINT, url_prefix=SWAGGER_URL)

# Define JSON schema
schema = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "gender": {"type": "array", "items": {"type": "string"}},
        "description": {"type": "string"},
        "type": {"type": "integer", "enum": [0, 1]},
        "producer": {"type": "string"},
        "studio": {"type": "string"},
    },
}


@app.route("/api/prediction", methods=["POST"])
def predict():
    json_data = request.get_json()
    try:
        validate(instance=json_data, schema=schema)
    except jsonschema.exceptions.ValidationError as err:
        return {"message": err.message}, 400
    # process the valid json_data here
    return jsonify({"result": random.randint(0, 5)})


if __name__ == "__main__":
    app.run(host="0.0.0.0")
