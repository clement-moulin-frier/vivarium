# vivarium
 
## Install

- Dependencies
  - python 3.9 (?)
  - [jax-md](https://github.com/jax-md/jax-md)
  - For the web interface
    - [panel](https://panel.holoviz.org/)
    - [param](https://param.holoviz.org/)
    - [grpc](https://grpc.io/docs/languages/python/quickstart/)
  - To control the server from a notebook
    - Jupyter Notebook

- Install
  - Once you have installed the dependencies above and cloned this repository, go to the base folder and run:
    - `pip install -e .`

- Execution
  - Run in a terminal
    - `python vivarium/simulator/grpc_server/simulator_server.py`
  - And launch the web interface from another terminal:
    - `panel serve vivarium/interface/panel_app.py --autoreload`
    - Once this command will have completed, it will output a URL looking like `http://localhost:5006/panel_app`, that you can open in your browser.
  - You will find explanations of the web interface and how to control the simulator programmatically in the Jupyter Notebook `vivarium/notebooks/quickstart_tutorial.ipynb`

- grpc compilation command line (normally only needed if modifying the .proto file for communication between server and controllers, e.g. the web interface):
```
python -m grpc_tools.protoc -I./vivarium/simulator/grpc_server/protos --python_out=./vivarium/simulator/grpc_server/ --pyi_out=./vivarium/simulator/grpc_server/ --grpc_python_out=./vivarium/simulator/grpc_server/ ./vivarium/simulator/grpc_server/protos/simulator.proto
```
