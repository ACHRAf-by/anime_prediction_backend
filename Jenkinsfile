pipeline {
    agent any
    
    stages {
        
        stage('Clone repo dev branch'){
            steps{
                git branch: 'dev', credentialsId: 'jenkins-backend', url: 'git@github.com:Atheros7/anime_list_backend.git'
                sh "ls"
            }
        }
    }
}
