FROM python:3.9-slim
WORKDIR /app
COPY . /app
RUN pip install mlflow dagshub pandas scikit-learn
CMD ["python", "modelling_tuning.py"]
