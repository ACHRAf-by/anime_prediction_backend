pipeline {
    agent any
    
    stages {
        
        stage('Clone'){
            steps{
                sh 'whoami'
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
        
        stage('Test') {
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
                        sh 'git merge --no-ff staging'
                        sshagent(credentials: ['jenkins-backend']){
                            sh 'git push origin main'
                        }
                    }
                    sh 'echo "Current Branch: $(git rev-parse --abbrev-ref HEAD)"'
                }
            }
        }
        
        stage('Docker') {
          steps {
            withCredentials([usernamePassword(credentialsId: 'dockerhub', usernameVariable: 'DOCKER_USER', passwordVariable: 'DOCKER_PASS')]) {
              script {
                def currentBranch = sh(returnStdout: true, script: 'git rev-parse --abbrev-ref HEAD').trim()
                if (currentBranch == 'main') {
                  env.DOCKER_USER = DOCKER_USER
                  env.DOCKER_PASS = DOCKER_PASS
                  sh 'docker login -u $DOCKER_USER -p $DOCKER_PASS'
                  sh 'docker build -t jeandevise/anime-backend:latest .'
                  sh 'docker push jeandevise/anime-backend:latest'
                } else {
                  echo "Skipping Docker build and push because current branch is ${currentBranch}"
                }
              }
            }
          }
        }
        
    }
}
