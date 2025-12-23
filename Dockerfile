FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-venv python3-pip curl bzip2 nano mc  \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel
RUN pip install cupy-cuda12x matplotlib jupyterlab numpy pandas pyarrow ipywidgets jupyterlab-widgets
WORKDIR /workspace
COPY libwwz/ /workspace/libwwz/
EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]
