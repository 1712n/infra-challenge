FROM pytorch/pytorch:latest

RUN mkdir /code

COPY ./requirements.txt /code/requirements.txt
COPY ./app.py /code/app.py
COPY ./models.py /code/models.py

WORKDIR /code
RUN pip install -r /code/requirements.txt

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]