FROM python:3.8-slim

RUN mkdir /streamlit

COPY requriements.txt /streamlit

WORKDIR /streamlit

RUN apt-get update && apt-get install -y python3-opencv

RUN pip install opencv-python

RUN pip install -r requriements.txt

COPY . /streamlit

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]