FROM python:3.7

RUN mkdir /app
WORKDIR /app
ADD . /app/

# Install production dependencies.
RUN pip install -r requirements.txt

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app

#EXPOSE 5000
#CMD ["python", "main.py"]
