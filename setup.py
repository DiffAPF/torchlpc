import setuptools
import os
import glob
import torch
from torch.utils.cpp_extension import (
    CppExtension,
    CUDAExtension,
    BuildExtension,
    CUDA_HOME,
)

library_name = "torchlpc"
VERSION = "0.7.2"
MAINTAINER = "Chin-Yun Yu"
EMAIL = "chin-yun.yu@qmul.ac.uk"


with open("README.md", "r") as fh:
    long_description = fh.read()


# if torch.__version__ >= "2.6.0":
#     py_limited_api = True
# else:
py_limited_api = False


def get_extensions():
    use_cuda = torch.cuda.is_available() and CUDA_HOME is not None
    use_openmp = torch.backends.openmp.is_available()
    extension = CUDAExtension if use_cuda else CppExtension

    extra_link_args = []
    extra_compile_args = {}
    if use_openmp:
        extra_compile_args["cxx"] = ["-fopenmp"]
        extra_link_args.append("-fopenmp")

    this_dir = os.path.abspath(os.path.dirname(__file__))
    extensions_dir = os.path.join(this_dir, library_name, "csrc")
    sources = list(glob.glob(os.path.join(extensions_dir, "*.cpp")))

    extensions_cuda_dir = os.path.join(extensions_dir, "cuda")
    cuda_sources = list(glob.glob(os.path.join(extensions_cuda_dir, "*.cu")))

    if use_cuda:
        sources += cuda_sources

    ext_modules = [
        extension(
            f"{library_name}._C",
            sources,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            py_limited_api=py_limited_api,
        )
    ]

    return ext_modules


setuptools.setup(
    name=library_name,
    version=VERSION,
    author=MAINTAINER,
    author_email=EMAIL,
    description="Fast, efficient, and differentiable time-varying LPC filtering in PyTorch.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DiffAPF/torchlpc",
    packages=["torchlpc"],
    install_requires=["torch>=2.0", "numpy", "numba"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX",
    ],
    license="MIT",
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp39"}} if py_limited_api else {},
)
