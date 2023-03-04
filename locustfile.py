from locust import HttpUser, task

class LoadTestUser(HttpUser):

    @task(10)
    def test_post(self):
        headers = { 'Content-Type': 'application/json'}
        data = {
            'Title': 'Ghost in the Shell',
            'Gender': ['Action', 'Adventure'],
            'Synopsis': 'In this post-cyberpunk iteration of a possible future, computer technology has advanced to the point that many members of the public possess cyberbrains, technology that allows them to interface their biological brain with various networks',
            'Type': "TV",
            'Producer': 'Koichi Yuri',
            'Studio': 'Production I.G',
            "Source": "Manga"
        }
        self.client.post('/api/prediction', headers=headers, json=data)

