# Dockerfile to build a flask app
# syntax=docker/dockerfile:1

FROM python:3

WORKDIR /usr/app

COPY . .
RUN pip3 install -r requirements.txt

EXPOSE 80

CMD [ "python", "app.py" ]
