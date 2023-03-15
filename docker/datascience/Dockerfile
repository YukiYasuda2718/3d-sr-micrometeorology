FROM jupyter/datascience-notebook:r-4.1.3

USER root

ENV HOME=/home/jovyan \
    LD_LIBRARY_PATH=/usr/local/lib/

ARG DEBIAN_FRONTEND="noninteractive"

EXPOSE 8888

WORKDIR /workspace

RUN useradd -M vscode && \
    chmod -R 777 ${HOME}

# The libraries below `less` are necessary to use `xgrads` and `opencv`
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ttf-mscorefonts-installer \
        less \
        build-essential \
        pkg-config \
        zlib1g-dev \
        libsqlite3-dev \
        sqlite3 \
        libcurl4-gnutls-dev \
        libtiff5-dev \
        libgl1-mesa-dev \
        libopencv-dev && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# proj and geos are necessary to use `xgrads`
# c.f.)
# http://hydro.iis.u-tokyo.ac.jp/~akira/page/Python/contents/plot/cartopy/installation.html
# https://qiita.com/XPT60/items/be4e3d9f9f58d17cf968
RUN mkdir ${HOME}/tmp && cd ${HOME}/tmp && \
    wget https://download.osgeo.org/proj/proj-8.2.1.tar.gz && \
    tar xvzf proj-8.2.1.tar.gz && \
    cd proj-8.2.1 && \
    ./configure && make && make install && cd .. && \
    wget https://download.osgeo.org/geos/geos-3.8.2.tar.bz2 && \
    tar xvf geos-3.8.2.tar.bz2 && \
    cd geos-3.8.2 && \
    ./configure && make && make install && cd .. && \
    cd ${HOME} && rm -rf ${HOME}/tmp

# Install libraries for data analysis
COPY Pipfile Pipfile.lock ${HOME}/

RUN cd ${HOME} && \
    pip install -U pip pipenv && \
    pipenv install --system --dev --ignore-pipfile && \
    rm -rf ${HOME}/.cache/pip ${HOME}/.cache/pipenv

# Copy JupyterLab formatter configs
COPY .jupyter/ ${HOME}/.jupyter/
