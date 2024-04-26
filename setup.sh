#!/bin/bash

if ! lsb_release -a | grep -q "Ubuntu 22.04"; then
  echo "This script is intended for Ubuntu 22.04. Proceed anyway? [y/n]"
  read -r response
  if [[ ! "$response" =~ ^[Yy]$ ]]; then
    exit 1
  fi
fi

if [ ! -f ./shared/workspace/urls_list.txt ]; then
  echo "https://www.gutenberg.org/cache/epub/514/pg514.txt" > ./shared/workspace/urls_list.txt
fi

#set debian as non-interactive
export DEBIAN_FRONTEND=noninteractive

# Check if python3-pip is installed
if ! command -v pip3 &> /dev/null
then
    echo "python3-pip is not installed. Installing..."
    sudo apt update
    sudo apt install -y python3-pip
else
    echo "python3-pip is already installed."
fi
echo "s1"

sudo apt install -y build-essential tmux jq curl wget git unzip

PATH=/home/blub0x/.local/bin:$PATH

#check output of nvidia-smi for cuda version 12.1, if not installed, install it
if ! nvidia-smi | grep -q "CUDA Version: 12.1"; then
  echo "CUDA 12.1 not installed. Installing..."
  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb -O cuda-keyring_1.0-1_all.deb
  sudo dpkg -i cuda-keyring_1.0-1_all.deb
  sudo apt-get update
  sudo apt-get -y install cuda
  rm cuda-keyring_1.0-1_all.deb
else
  echo "CUDA 12.1 already installed."
fi
echo "s2"

pip3 install -U pip

#-----------

for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do sudo apt-get remove $pkg; done

sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo "s3"

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update

sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

echo "s4"

sudo groupadd docker

sudo usermod -aG docker $USER

if [ -f /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg ]; then
  sudo rm /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
fi

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update

sudo apt-get install -y nvidia-container-toolkit

sudo nvidia-ctk runtime configure --runtime=docker

sudo systemctl restart docker

#docker build . -t langchain_ollama -f Dockerfile.langchain_ollama
docker build . -t langchain -f Dockerfile.langchain

pip3 install docker json-repair

sudo reboot