# vivarium
 
## Install

- Dependencies
  - python 3.9 (?)
  - jax-md
  - For the web interface
    - panel > 1.x
      - 18/05/23
        - Had to install it with `conda install -c conda-forge panel` (but v1.x was just released, just follow the doc now)
    - param
    - [grpc](https://grpc.io/docs/languages/python/quickstart/)
  - To control the server from a notebook
    - jupyterlab

- Install
  - `pip install -e .`
- Run
  - `python vivarium/simulator/grpc_server/simulator_server.py`
  - Web interface
    - `panel serve vivarium/interface/panel_app.py --autoreload`

- grpc compilation command line (normally only needed if modifying the .proto file for communication between server and controllers, e.g. the web interface):
```
python -m grpc_tools.protoc -I./vivarium/simulator/grpc_server/protos --python_out=./vivarium/simulator/grpc_server/ --pyi_out=./vivarium/simulator/grpc_server/ --grpc_python_out=./vivarium/simulator/grpc_server/ ./vivarium/simulator/grpc_server/protos/simulator.proto
```