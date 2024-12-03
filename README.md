# 🌱 Vivarium

**Vivarium** is framework for multi-agents simulations in Jax. It enables creating simple agents with two motors and two sensors, inspired by [Braitenberg Vehicles ](https://en.wikipedia.org/wiki/Braitenberg_vehicle), in a 2D rigid-body physics world. The physics engine is written in [Jax-MD](https://github.com/jax-md/jax-md). With Vivarium, you can run predefined or custom simulations, and interact with them in real time using a Web Interface or Jupyter notebooks. See our tutorials to get started with the project. 

![Vivarium demo](images/simulation.gif)

See a preliminary demo of the project on [this video](https://youtu.be/dnO-wo6Ns-8).

## 📥 Installation

1- Clone the repository:

```bash
git clone git@github.com:flowersteam/vivarium.git
cd vivarium/
```
2- (Optional) Create and activate a virtual environment:

```bash
python -m venv myvenv
source myvenv/bin/activate
```

3- Install the dependencies:

```bash
pip install -r requirements.txt
pip install -e . 
```

## 🚀 Usage

### Run the simulation in a server 🖥️

To run the simulation in a server, use the following command:

```bash
python3 scripts/run_server.py
```

We use [Hydra](https://hydra.cc/docs/intro/) to manage simulations data. By default, the simulation will use the parameters specified in the `default.yaml` scene file located in the `conf/scene` directory.

#### Using custom scene files 🌄

You can customize the initial simulation parameters by creating your own scene files in YAML format and placing them in the `conf/scene` directory. Scene files can specify parameters such as the number of objects, their size, or the colors, positions, and behaviors of agents for example. See Tutorial [Create a custom Scene](notebooks/tutorials/create_custom_scene_tutorial.md) for more information.

To use a custom scene file in your simulation, pass the `scene` option followed by the name of the scene file (without the `.yaml` extension) to the `run_server.py` script. For example, to run the `prey_predator_large` scene, use the following command:

```bash
python3 scripts/run_server.py scene=prey_predator_large
```

Any parameters not specified in the custom scene file will be inherited from the `default.yaml` scene.

### Interact with it from a web interface 🌐

When the server is started, you can launch a web interface from another terminal to observe the simulation and interact with it:

```bash
panel serve scripts/run_interface.py --autoreload
```

Once this command will have completed, it will output a URL looking like `http://localhost:5006/run_interface`. Just click on it, and it will open the web interface in your browser. From there you can start the simulation and play with it.


### Interact with it from a jupyter notebook 📓

You can also choose to control the simulator programmatically in Jupyter Notebook. We first recommend you to do the first tutorials listed below. Then you can select a notebook in the `notebooks/sessions` [directory](notebooks/sessions/README.md) and start playing with it ! If you choose to do so, you don't need to start the server and the interface manually. Instead, you can start and stop them with a custom command you can find in the notebooks.

## 📚 Tutorials

To help you get started and explore the project, we provide a set of Jupyter notebook tutorials located in the `notebooks/tutorials` [directory](notebooks/tutorials/README.md). These tutorials cover various aspects of the project, from using the graphical interface to interacting with simulations and understanding the backend.

- **Web Interface Tutorial**: Begin with the [web interface tutorial](notebooks/tutorials/web_interface_tutorial.md) to gain a basic understanding of the project and learn how to use the graphical interface.
- **Quickstart Tutorial**: To learn how to interact with a simulation from a Jupyter notebook, follow the [quickstart tutorial](notebooks/tutorials/quickstart_tutorial.ipynb). This tutorial will guide you through creating, running, and manipulating simulations within a notebook environment.
- **Create a custom Scene**: If you want to create your own simulations with custom parameters, check out the [create a custom scene tutorial](notebooks/tutorials/create_custom_scene_tutorial.md). This tutorial will show you how to create and use custom scene files to define the initial parameters of your simulations. 
- **Simulator tutorial**: For a deeper understanding of the simulator backend and its capabilities, check out the [simulator tutorial](notebooks/tutorials/simulator_tutorial.ipynb). This tutorial provides insights into the underlying mechanics of the simulator and demonstrates how to leverage its features for advanced use cases

## 🛠 Development

### gRPC Configuration 🔄

The projecte uses gRPC to communicate between server and clients. If you made any changes in the .proto file, you will need to recompile the gRPC files. Here is the command line instruction to do so:


```bash
python -m grpc_tools.protoc -I./vivarium/simulator/grpc_server/protos --python_out=./vivarium/simulator/grpc_server/ --pyi_out=./vivarium/simulator/grpc_server/ --grpc_python_out=./vivarium/simulator/grpc_server/ ./vivarium/simulator/grpc_server/protos/simulator.proto
```

### Running Automated Tests 🧪 

If you want to add tests for your local changes, you can write them in the `tests/` directory. Make sure that the name or your files and test functions start with "test". You can then run the following command in the root of the directory to launch them :

```bash
pytest
```

## Acknowledgments

The main contributors of this repository are Corentin Léger and Clément Moullin-Frier from the Flowers team at Inria, with participation of Martial Marzloff. CMF initiated the code base architecture in 2023 and CL was the main developper in 2024. CL was funded by the [French National Research Agency](https://anr.fr/), project ECOCURL, Grant ANR-20-CE23-0006. 
