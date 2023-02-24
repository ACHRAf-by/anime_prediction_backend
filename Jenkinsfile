pipeline {
    agent any
    
    stages {
        
        stage('Clone repo dev branch'){
            steps{
                git branch: 'dev', credentialsId: 'jenkins-backend', url: 'git@github.com:Atheros7/anime_list_backend.git'
                sh "git branch -D staging"
                sh "git checkout -b staging"                
            }
        }
        
        stage('Build app') {
            steps {
                dir("CI_Jenkins"){
                    echo 'pip install -r requirements.txt'
                    sh "pip install -r requirements.txt"
                }
            }
            
        }
    }
}
