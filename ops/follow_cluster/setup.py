from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='follow_cluster_cuda',
    ext_modules=[
        CUDAExtension('follow_cluster_cuda', [
            'src/follow_cluster_cuda.cpp',
            'src/follow_cluster_kernel.cu',
        ]),
    ],
    cmdclass={'build_ext': BuildExtension})
