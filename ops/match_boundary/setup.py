from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='match_boundary_cuda',
    ext_modules=[
        CUDAExtension('match_boundary_cuda', [
            'src/match_boundary_cuda.cpp',
            'src/match_boundary_kernel.cu',
        ]),
    ],
    cmdclass={'build_ext': BuildExtension})
