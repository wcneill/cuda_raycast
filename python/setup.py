from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='raycast',
      ext_modules=
      [
          Extension(
              name="raycast", 
              sources=['extension.cpp'],
              
              include_dirs= 
                  cpp_extension.include_paths() + 
                  [
                        '../external/libtorch/include', 
                        "../external/libtorch/include/torch/csrc/api/include",
                        "../libray/include",
                        'C:\\Users\\wesle\\anaconda3\\lib\\site-packages\\torch\\include',
                        'C:\\Users\\wesle\\anaconda3\\lib\\site-packages\\torch\\include\\torch\\csrc\\api\\include', 
                        'C:\\Users\\wesle\\anaconda3\\lib\\site-packages\\torch\\include\\TH',
                        'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.6\\include'
                  ],

              library_dirs=[
                  "../build/libray/include/Release/", 
                  "../external/libtorch/lib",
                  "C:\\Users\\wesle\\anaconda3\\Lib\\site-packages\\torch\\lib",
                  'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.6\\lib/x64'],

              libraries=["libray", 'c10', 'torch', 'torch_cpu', 
                         'torch_python', 'cudart', 'c10_cuda', 
                         'torch_cuda_cu', 'torch_cuda_cpp']) 
      ],
      cmdclass={'build_ext': cpp_extension.BuildExtension})