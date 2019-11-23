FROM python:3.7

WORKDIR /usr/src/app
COPY . .

RUN pip install pipenv
RUN pipenv run pip install pip==18.0
RUN pipenv install --system

CMD ['python3', 'rnn.py']