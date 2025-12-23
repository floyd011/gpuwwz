docker run --rm --name cuptest --gpus all -d  -p 8888:8888 -v $(pwd):/workspace -e JUPYTER_TOKEN="08042011" libcupy 
