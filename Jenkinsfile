pipeline {
    agent any
    
    stages {
        
        stage('Clone'){
            steps{
                git branch: 'dev', credentialsId: 'jenkins-backend', url: 'git@github.com:Atheros7/anime_list_backend.git'
                sh "git branch -D staging || true"
                sh "git checkout -b staging"                
            }
        }
        
        stage('Build') {
            steps {
                sh "pip install -r requirements.txt"  
            }      
        }
        
        stage('test from github') {
            steps {
                sh "python3 -m unittest"
            }
        }
        
        stage('Merge') {
            steps {
                script {
                    def merge = input(
                        message: 'Do you want to merge with the main branch?',
                        parameters: [
                            booleanParam(defaultValue: false, description: '', name: 'merge')
                        ]
                    )
                    
                    if (merge) {
                        sh 'git checkout main'
                        sh 'git merge --no-ff origin/staging'
                        sh 'git push origin main'
                    }
                }
            }
        }
        
        stage('Docker') {
            when {
                branch 'main'
            }
            steps {
                sh 'docker login -u=${dockerhub_USR} -p=${dockerhub_PSW}'
                sh 'docker build -t your-dockerhub-username/your-app-name:latest .'
                sh 'docker push your-dockerhub-username/your-app-name:latest'
            }
        }

    }
}
