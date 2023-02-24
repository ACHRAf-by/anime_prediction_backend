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
        
        stage('Build') {
            steps {
                sh "pip install -r requirements.txt"  
            }      
        }
        
        stage('test from github') {
            steps {
                sh "pyhton -m unittest"
            }
        }
    }
}
