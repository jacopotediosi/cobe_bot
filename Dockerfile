FROM alpine:latest

COPY requirements.txt .
COPY /cobe /cobe
COPY /src /src

RUN apk update && \
    apk upgrade && \
    apk add python3 python3-dev py3-pip build-base mariadb-dev && \
    pip3 install -r requirements.txt && \
    cd /cobe && \
    python3 setup.py build && \
    python3 setup.py install && \
    rm -rf /var/cache/apk/* /cobe requirements.txt

WORKDIR /src

CMD ["python3", "main.py"]
