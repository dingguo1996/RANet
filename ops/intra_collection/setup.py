from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='intra_collection_cuda',
    ext_modules=[
        CUDAExtension('intra_collection_cuda', [
            'src/intra_collection_cuda.cpp',
            'src/intra_collection_kernel.cu',
        ]),
    ],
    cmdclass={'build_ext': BuildExtension})
