# syntax=docker/dockerfile:1.4

FROM --platform=$BUILDPLATFORM python:3.9-alpine AS builder
EXPOSE 8000
WORKDIR /RAG_QA
COPY requirements.txt /RAG_QA
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install --no-cache-dir gunicorn[gevent]
COPY . .
CMD exec gunicorn --bind 0.0.0.0:8000 --workers 1 --threads 8 --timeout 0 RAG_QA.wsgi:application




#
#
#
## syntax=docker/dockerfile:1.4
#
#FROM --platform=$BUILDPLATFORM python:3.9-alpine AS builder
#EXPOSE 8000
#WORKDIR /RAG_QA
#COPY requirements.txt /RAG_QA
#RUN /usr/local/bin/python -m pip install --upgrade pip
#RUN pip3 install -r requirements.txt --no-cache-dir
#COPY . /RAG_QA
#RUN if [ ! -f db.sqlite3 ]; then \
#    python3 manage.py migrate; \
#    python3 manage.py makemigrations; \
#    python3 manage.py migrate; \
#    python3 manage.py makemigrations annotations; \
#    python3 manage.py migrate annotations; \
#    python3 manage.py makemigrations dialogue; \
#    python3 manage.py migrate dialogue; \
#    python3 manage.py makemigrations users; \
#    python3 manage.py migrate users; \
#    python3 manage.py createsuperuser --noinput --username admin --email; \
#    fi
#ENTRYPOINT ["python3"]
#CMD ["manage.py", "runserver", "0.0.0.0:8000"]
#
