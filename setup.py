import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='PyNNN',
    version='0.0.1',
    author="Ahmed Khaled",
    author_email="ahmedkhaled11119999@gmail.com",
    description='A neural network framework based on numpy only',
    packages=setuptools.find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/EslamAsfour/PyNNN",
    classifiers=[
        "License :: OSI Approved :: MIT License",
    ],
    install_requires=['numpy'],
    python_requires='>=3.6',
    )
