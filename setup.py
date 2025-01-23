import setuptools
from torch.utils import cpp_extension

NAME = "torchlpc"
VERSION = "0.7.dev"
MAINTAINER = "Chin-Yun Yu"
EMAIL = "chin-yun.yu@qmul.ac.uk"


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name=NAME,
    version=VERSION,
    author=MAINTAINER,
    author_email=EMAIL,
    description="Fast, efficient, and differentiable time-varying LPC filtering in PyTorch.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yoyololicon/torchlpc",
    packages=["torchlpc"],
    install_requires=["torch>=2.0", "numpy", "numba"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    ext_modules=[
        cpp_extension.CppExtension("torchlpc._C", ["torchlpc/csrc/scan_cpu.cpp"])
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
