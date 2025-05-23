from setuptools import find_packages, setup

setup(
    name="lcmr-ext",
    version="0.1.0",
    description="...",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9.0",
    install_requires=[
        "multiprocess",
        "more-itertools",
        "imageio",
        "omegaconf",
        "torchmetrics",
        "open_clip_torch",
        "lpips",
        #'opencv-python'
    ],
)
