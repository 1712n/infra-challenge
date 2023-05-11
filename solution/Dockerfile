FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

COPY ./app.py /code/app.py

RUN pip install -r requirements.txt 

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]gi