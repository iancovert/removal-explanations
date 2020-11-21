import setuptools

setuptools.setup(
    name="removal-explanations",
    version="0.0.1",
    author="Ian Covert",
    author_email="icovert@cs.washington.edu",
    description="For explaining black-box models using removal-based explanations.",
    long_description="""
        Removal-based explanations are a class of model explanation method that 
        unifies many existing approaches (e.g., SHAP, LIME, Meaningful 
        Perturbations, permutation tests). This repository expresses many of 
        these methods using a lightweight, modular implementation. 
    """,
    long_description_content_type="text/markdown",
    url="https://github.com/iancovert/removal-explanations/",
    packages=['rexplain'],
    install_requires=[
        'numpy',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering"
    ],
    python_requires='>=3.6',
)
