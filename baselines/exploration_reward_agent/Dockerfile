FROM debian:buster

MAINTAINER TextWorld Team <textworld@microsoft.com>

RUN apt-get update -qy
RUN apt-get install -qy build-essential uuid-dev libffi-dev python3.6-dev curl git docker.io
RUN apt-get install -qy python3-pip wget graphviz
RUN pip3 install docker
RUN pip3 install textworld==1.0.0
RUN pip3 install spacy
RUN python3 -m spacy download en
RUN pip3 install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp36-cp36m-linux_x86_64.whl
RUN pip3 install torchvision
