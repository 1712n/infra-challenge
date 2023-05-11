FROM pytorch/pytorch:latest

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

COPY ./app.py /code/app.py

RUN pip install -r requirements.txt 

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]gi
