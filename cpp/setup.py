import setuptools
from torch.utils import cpp_extension

setup(name='raycast',
      ext_modules=[cpp_extension.CppExtension('raycast', ['raycast.cpp, raycast_kernel.cu'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
