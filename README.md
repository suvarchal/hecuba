# Hecuba ![](https://travis-ci.org/bsc-dd/hecuba.svg?branch=master) [![codecov](https://codecov.io/gh/bsc-dd/hecuba/branch/master/graph/badge.svg)](https://codecov.io/gh/bsc-dd/hecuba)
Non-relational databases are nowadays a common solution when dealing with a huge data set and massive query workload. These systems have been redesigned from scratch to achieve scalability and availability at the cost of providing only a reduced set of low-level functionality, thus forcing the client application to take care of complex logic. As a solution, our research group developed **Hecuba**, a set of tools and interfaces, which aims to facilitate programmers with an efficient and easy interaction with non-relational technologies.

## Installation procedure

### Software requisites:

+ GCC >= 5.4.0
+ CMake >= 3.14
+ Python >= 3.6 development version installed.

### Dependencies:
+ [Apache Cassandra](https://github.com/apache/cassandra) >= 4.0
+ [Kafka](https://kafka.apache.org) >= 2.13

### Python dependencies
+ numpy library >= 1.16
+ Cassandra driver for python >= 3.7.1
+ nose >= 1.3.7
+ ccm
+ mock

### C++ dependencies (Auto-downloaded during the installation process)
+ [Cassandra driver for C++](https://github.com/datastax/cpp-driver) >= 2.14.1
+ [Intel Threading Building Blocks (libTBB)](https://github.com/01org/tbb) >= 2020.0
+ [libuv](https://github.com/libuv/libuv) >= 1.11.0
+ [Apache Arrow](https://github.com/apache/arrow) >= 0.15.1
+ [yaml-cpp](https://github.com/jbeder/yaml-cpp) >= 0.7.0
+ [librdkafka](https://github.com/edenhill/librdkafka) >= 1.9.2

### Automatic install

To ease Hecuba installation we have prepared a wheel package with all the required dependencies, and installable through the 'pip' tool:

```bash
pip install hecuba
```

This will install in the default installation directories, typically the system, but you can use the '--user' flag or use a virtual environment to change the target directory.

This installation only supports X86_64 architectures and has used GCC to generate the libraries.

## Manual installation

The first step is to download the code.

```bash
# Clone the repository
git clone https://github.com/bsc-dd/hecuba.git hecuba_repo
cd hecuba_repo
```

Install the python dependencies using:
```bash
pip install -r requirements.txt
```

Then it is only necessary to run the `setup.py` Python script, which performs all the steps to compile and install Hecuba in the system. Notice that Hecuba is composed by Python code and C++ code. The `setup.py` script takes care of both, on the one hand, compiles the C++ Hecuba code and its dependencies and installs their C++ header files and generated libraries; and on the other hand, installs the Python package.
The parameters passed to the `setup.py` script indicates what to do (build or install) and  where to install Hecuba and its dependencies.

### Compilation

Use the following command to compile Hecuba C++ library, its dependencies and the Python wrapper:

```bash
python setup.py build

```

At first the dependencies are checked in the machine, looking at the directories stated in `LD_LIBRARY_PATH` and the hecuba's compilation directory (`build/lib` by default). Any unmet dependency will be downloaded, compiled and installed in the compilation directory (`build/lib`).

You may use the `--c_binding` flag to set the directory where to search for libraries and store the new generated libraries.

This procedure launches CMake to build the dependencies, which may take some time.


### Installation
Once compiled, you may install the Python package using one of the following commands:

```bash
# (1) To install hecuba to the default system directory
python setup.py install
# (2) Install to user space, under $HOME/.local
python setup.py install --user
# (3) Install to a user-defined path $CUSTOM_PATH
python setup.py install --prefix=$CUSTOM_PATH
```

Where the `--user` and `--prefix` flags are used to define different directories for the Python package.
The target directory for the Python package can be the default system directory (1), the user space (`$HOME/.local`) (2) or a custom path (3).

Warning: Be sure that the `PYTHONPATH` variable contain the path to the Hecuba Python package in the case (3).

Warning: The python wrapper stores an RPATH for the dependencies, therefore the used/generated libraries during compilation phase must always be accessible at the same path (they can not be moved or deleted).


#### Compilation problems resolution
If CMake selects a non-compliant compiler, it can be explicitly selected by defining the environment variables `CC=/custom/path/gcc` and `CXX=/custom/path/g++`. Then remove the `hecuba_core/build` folder (where all the CMake cache files an generated object files reside) and restart the installation process.


### Auto-downloading process

Before starting the compilation, the installation procedure checks if the required libraries are already in the directories specified as the compilation directories or accessible via `LD_LIBRARY_PATH`. If they are not there, then it checks if their source code is in the directory `hecuba_core/dependencies`. If it is not there, then it downloads its source code.

Note: If you want to install Hecuba on a computer without internet access, first make an initial installation on a machine that has internet access and then copy the files under `hecuba_core/dependencies` to the remote computer under the same directory.



### Install the Hecuba core only

If you need just the C++ interface of Hecuba and want to skip the Python installation you can just build the C++ side using the following comands:

```bash
cmake -Hhecuba_core -Bbuild -DC_BINDING_INSTALL_PREFIX=$HECUBA_LIBS_PATH
make -C build
```

This will install under the `HECUBA_LIBS_PATH/lib` folder the C++ libraries and under `HECUBA_LIBS_PATH/include` the headers.


## Instructions to execute with Hecuba:

Please, refer to the [Hecuba manual](https://github.com/bsc-dd/hecuba/wiki/1:-User-Manual) for the execution instructions.


## LICENSING 

Copyright 2017 Barcelona Supercomputing Center

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
