from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='raycast_distance',
      ext_modules=[cpp_extension.CUDAExtension('raycast_distance', ['cpp/raycast.cpp', 'cpp/raycast_cuda.cu'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})