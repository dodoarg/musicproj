FROM python:3.9.7

RUN adduser --disabled-password --gecos '' ml-api-user

WORKDIR /opt/song-popularity-api

ADD ./song-popularity-api /opt/song-popularity-api/
RUN pip install --upgrade pip
RUN pip install -r /opt/song-popularity-api/requirements.txt

RUN chmod +x /opt/song-popularity-api/run.sh
RUN chown -R ml-api-user:ml-api-user ./

USER ml-api-user

EXPOSE 8001

CMD ["bash", "./run.sh"]