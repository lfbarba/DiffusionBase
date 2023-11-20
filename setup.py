from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'Diffusion model using Diffusers'
LONG_DESCRIPTION = ''

# Setting up
setup(
    # the name must match the folder name 'verysimplemodule'
    name="diffusion",
    version=VERSION,
    author="Luis Barba",
    author_email="<youremail@email.com>",
    description=DESCRIPTION,
    packages=find_packages(),
    install_requires=[],  # add any additional packages that
)