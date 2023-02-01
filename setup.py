import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="charm",
    version="0.0.1",
    author="Todd Morrill",
    author_email="tm3229@columbia.edu",
    description="A package for managing CHARM data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=['charm'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=[
       "pandas",
   ]
)