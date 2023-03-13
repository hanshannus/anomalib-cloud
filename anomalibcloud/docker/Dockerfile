FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04
# install dependencies from apt
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1-mesa-glx git
# install python packages
RUN pip install --upgrade pip
RUN pip install azure-ai-ml mldesigner azureml-mlflow wandb \
    https://github.com/hanshannus/anomalib-tools/releases/download/v0.1.2/anomalibtools-0.1.2-py3-none-any.whl
