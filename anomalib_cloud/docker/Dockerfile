FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04
# install dependencies from apt
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1-mesa-glx git
# install python packages
RUN pip install --upgrade pip
RUN pip install azure-ai-ml mldesigner azureml-mlflow wandb \
    git+https://github.com/hanshannus/anomalib-tools.git@main
