# Containers

**Jetson:** use dusty-nv's jetson-containers rather than writing Dockerfiles yourself:

    git clone https://github.com/dusty-nv/jetson-containers && bash jetson-containers/install.sh
    jetson-containers run -v $(pwd):/workspace $(autotag l4t-pytorch)

`autotag` picks an image matching your JetPack/L4T version. For ROS 2, look for the
Humble images in the same project. Inside the container: `pip install -r requirements-jetson.txt && pip install -e .`

**Training machine:** a plain Python venv with `requirements-train.txt` is enough. Record
the exact versions (`pip freeze > docs/train_env.txt`) once your baseline works.
