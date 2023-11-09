from setuptools import setup, find_packages

setup(
    name="worm-align",
    version="0.1",
    packages=find_packages(),
    author="Alicia Lu",
    author_email="lualicia88@gmail.com",
    install_requires=["numpy"],
    license="MIT",
    description="a neural network `for registering confocal calcium imaging frames of C. elegans",
    url="https://github.com/flavell-lab/worm-align"
)
