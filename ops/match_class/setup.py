from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='match_class_cuda',
    ext_modules=[
        CUDAExtension('match_class_cuda', [
            'src/match_class_cuda.cpp',
            'src/match_class_kernel.cu',
        ]),
    ],
    cmdclass={'build_ext': BuildExtension})
