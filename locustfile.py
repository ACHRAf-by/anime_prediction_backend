from locust import HttpUser, task

class LoadTestUser(HttpUser):

    @task(10)
    def test_post(self):
        headers = { 'Content-Type': 'application/json'}
        data = {
            'title': 'Ghost in the Shell',
            'gender': ['Action'],
            'description': 'In this post-cyberpunk iteration of a possible future, computer technology has advanced to the point that many members of the public possess cyberbrains, technology that allows them to interface their biological brain with various networks',
            'type': 0,
            'producer': 'Koichi Yuri',
            'studio': 'Production I.G'
        }
        self.client.post('/api/prediction', headers=headers, json=data)

