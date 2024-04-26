#determine absolute path of this file
THIS_FILE_PATH=$(realpath $0)
THIS_FILE_DIR=$(dirname $THIS_FILE_PATH)
SHARED_DIR_PATH=$THIS_FILE_DIR/shared

sudo nvidia-smi -pl 200

docker run --runtime=nvidia --rm -it \
    -v $SHARED_DIR_PATH/dot_cache:/root/.cache \
    -v $SHARED_DIR_PATH/workspace:/root/workspace \
    -v $SHARED_DIR_PATH/libs:/root/libs \
    langchain


#   -v $SHARED_DIR_PATH/dot_ollama:/root/.ollama \
#    langchain_ollama