FROM ubuntu:20.04

WORKDIR /home
COPY . ./

RUN bash /home/Miniconda3-py39_24.5.0-0-Linux-x86_64.sh -b -p /miniconda3 \
    && rm /home/Miniconda3-py39_24.5.0-0-Linux-x86_64.sh

ENV PATH=/miniconda3/bin:$PATH
ENV LANG="en_US.UTF-8"

RUN conda install -c conda-forge GDAL && \
  conda install -c conda-forge geopandas==0.13.2 && \
    conda install -c conda-forge shapely==2.0.1 && \
    pip install -r base.requirements.txt

CMD ["python3","main.py"]