FROM python:3.6.9-buster
MAINTAINER Andrea Ferretti

ENV APP_ROOT=/opt/charade
WORKDIR ${APP_ROOT}

RUN apt-get update
RUN apt-get -y install python-pip
RUN pip install pipenv

ENV PIP_DEFAULT_TIMEOUT=6000 PIPENV_TIMEOUT=6000
ENV USER_NAME=charade USER_UID=10001
ENV HOME=${APP_ROOT}
COPY Pipfile Pipfile
COPY Pipfile-linux.lock Pipfile.lock
RUN pipenv install --system --deploy

RUN python -m spacy download it
RUN python -m spacy download en
RUN python -m spacy download de
RUN python -m nltk.downloader averaged_perceptron_tagger
RUN python -m nltk.downloader maxent_ne_chunker
RUN python -m nltk.downloader words

RUN mkdir -p ${APP_ROOT}/bin && \
    chmod -R ug+x ${APP_ROOT}/bin && sync && \
    useradd -l -u ${USER_UID} -r -g 0 -d ${APP_ROOT} -s /sbin/nologin -c "${USER_NAME} user" ${USER_NAME} && \
    chown -R ${USER_UID}:0 ${APP_ROOT} && \
    chmod -R g=u ${APP_ROOT}

COPY resources ./resources
COPY app ./app
COPY src ./src

CMD uwsgi --uid ${USER_UID} --gid ${USER_UID} --socket 0.0.0.0:9000 \
  --protocol=http --processes 1 --stats 0.0.0.0:9001 --stats-http \
  --module src.prod --callable app --lazy-apps