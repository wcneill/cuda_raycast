# CUDA Raycasting Demo

I started this as a little learner project in order to teach myself some skills that pushed a little bit past
what was available in the beginner tutorials:

- Programming using CUDA/C++.
- Configure a CMake Project to create both libraries and executables.
- Create Python bindings for an existing library. 

## Building C++
In order to build this library yourself, you will have to download and unzip `libtorch` from https://pytorch.org/get-started/locally/. 
Once unzipped, move the `libtorch/` directory to a new folder called `external`, which should be at the top level of this repo. From 
there you should be ready to go using CMake

1. `mkdir build`
2. `cd build`
3. `cmake ..`
4. `cmake --build . --config <Debug or Release>`

## Running C++ Test
Simply search the generated `build/test/<Debug or Release>/` and run `cuda_test.exe` from the command line. 

## Building Python Bindings
For this you will need to have the python `torch` library installed in your environment in addition to `libtorch` mentioned in the last
section **Building C++**

1. Go to `cuda_raycast/python/setup.py` and edit the directories under `include_dirs` and `library_dirs` to match your installation of `torch`. 
2. On the command line, navigate to `cuda_raycast/python`
3. Run `python setup.py install`.

## Running Python Test
On the commandline run `python cuda_raycast/python/ray.py`. 
