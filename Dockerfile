FROM rs-toolbox:base

WORKDIR /home

COPY . ./

CMD ["uvicorn","main:app", "--host", "0.0.0.0", "--port", "8080"]