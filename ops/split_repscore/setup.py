from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='split_repscore_cuda',
    ext_modules=[
        CUDAExtension('split_repscore_cuda', [
            'src/split_repscore_cuda.cpp',
            'src/split_repscore_kernel.cu',
        ]),
    ],
    cmdclass={'build_ext': BuildExtension})
