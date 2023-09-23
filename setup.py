import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="indobert_embedding",
    version="0.0.1",
    author="Robin Syihab",
    description="Text embedding encoder using IndoBERT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    py_modules=["indobert_embedding"],
    package_dir={'': 'indobert_embedding/src'},
    install_requires=[
        "transformers==4.33.2",
        "accelerate==0.20.3",
        "numpy==1.24.2",
        "torch==2.0.0",
        "torchmetrics==0.11.4"
    ]
)
