#!/bin/bash
xhost +local: # habilita al host para que el contenedor docker se conecte al se>

# Comprobar si se proporcionó un nombre de contenedor
if [ -z "$1" ]; then
  echo "Error: Debes proporcionar un nombre para el contenedor."
  exit 1
fi

# Obtener los contenedores con el nombre dado
CONTAINER_NAME=$(docker ps -a -f "name=$1" --format '{{.Names}}')

# Comprobar si el contenedor ya existe
if [ -n "$CONTAINER_NAME" ] && [ "$CONTAINER_NAME" == "$1" ]; then
  echo "Reanudando contenedor existente $1..."
  docker start $1
  docker attach $1
else
  echo "Creando nuevo contenedor $1..." #llm_network
  docker run -it --rm --gpus all \
    --network host \
    -e DISPLAY=${DISPLAY} \
    -e "HF_HOME=/home/$USER/ia_cache/" \
    -e "LLAMA_INDEX_CACHE_DIR=/home/$USER/ia_cache/" \
    -e "TRANSFORMERS_CACHE=/home/$USER/ia_cache/" \
    -e "HUGGINGFACE_HUB_CACHE=/home/$USER/ia_cache/" \
    -e "TOKENIZERS_PARALLELISM=true" \
    --add-host=hostname:127.0.0.1 \
    -e USER_PASSWORD=dev.2m \
    -e SSH_PORT=2222 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v ${PWD}/.ssh:/root/.ssh:ro \
    -v ${PWD}:/home/$USER \
    -w /home/$USER \
    --name $1 \
    --shm-size=840m \
    -e HOST_USERNAME=$USER \
    --cap-add SYS_ADMIN \
    -e HOST_UID=$(id -u) \
    -e HOST_GID=$(id -g) \
    cuda12_1_ssh /bin/bash
fi
