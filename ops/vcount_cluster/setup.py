from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='vcount_cluster_cuda',
    ext_modules=[
        CUDAExtension('vcount_cluster_cuda', [
            'src/vcount_cluster_cuda.cpp',
            'src/vcount_cluster_kernel.cu',
        ]),
    ],
    cmdclass={'build_ext': BuildExtension})
