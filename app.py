# Dependencies
from flask import Flask, request, jsonify
import pandas as pd
import jsonschema
from jsonschema import validate
from flask_cors import CORS
from flask_swagger_ui import get_swaggerui_blueprint
from prometheus_flask_exporter import PrometheusMetrics
from process_input import process_input


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
        "Title": {"type": "string"},
        "Gender": {"type": "array", "items": {"type": "string"}},
        "Description": {"type": "string"},
        "Type": {"type": "string"},
        "Producer": {"type": "string"},
        "Studio": {"type": "string"},
        "Source": {"type": "string"},
    },
}


@app.route("/api/prediction", methods=["POST"])
def predict():
    json_data = request.get_json()
    try:
        validate(instance=json_data, schema=schema)
    except jsonschema.exceptions.ValidationError as err:
        return {"message": err.message}, 400
    
    process_input(pd.DataFrame([json_data]))

    # process the valid json_data here
    return jsonify({"result": process_input(pd.DataFrame([json_data]))})


if __name__ == "__main__":
    app.run(host="0.0.0.0")
