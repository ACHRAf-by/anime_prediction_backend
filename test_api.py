import unittest
from flask import json
from app import app

class TestPredict(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.valid_data = {
            "title": "My Title",
            "gender": ["Action", "Adventure"],
            "description": "My description",
            "type": 0,
            "producer": "My Producer",
            "studio": "My Studio"
        }
        self.invalid_data = {
            "title": "My Title",
            "gender": "Invalid Gender",
            "description": "My description",
            "type": 0,
            "producer": "My Producer",
            "studio": "My Studio"
        }

    def test_valid_predict(self):
        response = self.app.post('/api/prediction', data=json.dumps(self.valid_data), content_type='application/json')
        self.assertEqual(response.status_code, 200)

    def test_invalid_predict(self):
        response = self.app.post('/api/prediction', data=json.dumps(self.invalid_data), content_type='application/json')
        self.assertEqual(response.status_code, 400)

if __name__ == '__main__':
    unittest.main()
