from glob import glob
from setuptools import setup

setup(
    name="esa_ros",
    version="0.1.0",
    packages=["esa_ros"],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/esa_ros"]),
        ("share/esa_ros", ["package.xml"]),
        ("share/esa_ros/launch", glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    entry_points={"console_scripts": ["detector_node = esa_ros.detector_node:main"]},
)
