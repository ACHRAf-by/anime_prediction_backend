import unittest
from flask import json
from app import app

class TestPredict(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.valid_data = {
            "Title": "My Title",
            "Gender": ["Action", "Adventure"],
            "Synopsis": "My description",
            "Type": "0",
            "Producer": "My Producer",
            "Studio": "My Studio",
            "Source": 1
        }
        self.invalid_data = {
            "Title": "My Title",
            "Gender": "Invalid Gender",
            "Synopsis": "My description",
            "Type": "0",
            "Producer": "My Producer",
            "Studio": "My Studio",
            "Source": "TV"
        }

    def test_valid_predict(self):
        response = self.app.post('/api/prediction', data=json.dumps(self.valid_data), content_type='application/json')
        self.assertEqual(response.status_code, 200)

    def test_invalid_predict(self):
        response = self.app.post('/api/prediction', data=json.dumps(self.invalid_data), content_type='application/json')
        self.assertEqual(response.status_code, 400)

if __name__ == '__main__':
    unittest.main()
