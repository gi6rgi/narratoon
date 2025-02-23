#!/bin/bash
set -e

if [ ! -d "Wav2Lip" ]; then
    echo "Cloning the wav2lip repository..."
    git clone https://github.com/Rudrabha/Wav2Lip.git
else
    echo "wav2lip repository already exists, updating..."
    cd Wav2Lip && git pull && cd ..
fi

WEIGHTS_DIR="Wav2Lip/face_detection/detection/sfd"
WEIGHTS_FILE="s3fd.pth"
WEIGHTS_URL="https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth"

if [ ! -f "${WEIGHTS_DIR}/${WEIGHTS_FILE}" ]; then
    echo "Downloading pre-trained weights..."
    mkdir -p "${WEIGHTS_DIR}"
    # Using wget to download the weights
    wget -O "${WEIGHTS_DIR}/${WEIGHTS_FILE}" "${WEIGHTS_URL}"
else
    echo "Weights file already exists."
fi

GAN_WEIGHTS_DIR="Wav2lip/checkpoints"
GAN_WEIGHTS_FILE="wav2lip_gan.pth"
GAN_WEIGHTS_URL="https://github.com/justinjohn0306/Wav2Lip/releases/download/models/wav2lip_gan.pth"

if [ ! -f "${GAN_WEIGHTS_DIR}/${GAN_WEIGHTS_FILE}" ]; then
    echo "Downloading GAN pre-trained weights..."
    mkdir -p "${GAN_WEIGHTS_DIR}"
    wget -O "${GAN_WEIGHTS_DIR}/${GAN_WEIGHTS_FILE}" "${GAN_WEIGHTS_URL}"
else
    echo "GAN weights file already exists."
fi

echo "Installing wav2lip dependencies..."
cd Wav2Lip
pip install -r requirements.txt
cd ..

echo "Checking for Poetry installation..."
if ! command -v poetry &> /dev/null; then
    echo "Poetry not found, installing..."
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "Poetry is already installed."
fi

echo "Installing project dependencies using Poetry..."
poetry install

echo "All steps completed successfully!"
