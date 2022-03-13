FROM debian

# Install python
RUN apt-get update -y && apt-get install -y python3 python3-pip python3-setuptools zlib1g-dev libjpeg-dev
RUN pip3 install requests \
    requests-toolbelt \
    cachecontrol \
    six \
    future \
    shapely \
    numpy \
    opencv-python-headless \
    Pillow==6.2.2

RUN apt-get install -y git

# Install Cytomine python client
RUN git clone https://github.com/cytomine/Cytomine-python-client.git && \
    cd Cytomine-python-client && \
    git checkout v2.2.0 && \
    python3 setup.py build && \
    python3 setup.py install

# Add requirements and install them
ADD requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Install libvips
RUN apt-get install -y libvips

# Add files
ADD assets /assets/
ADD predict.py /app/predict.py
ADD run.py /app/run.py

ENTRYPOINT [ "python3", "/app/run.py" ]

