from setuptools import setup, find_packages


setup(
    name="mega",
    version="0.9.0",
    description="...",
    author="Will Roper",
    author_email="w.roper@sussex.ac.uk",
    url="https://github.com/WillJRoper/mega",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.14.5",
        "scipy>=1.7",

    ],
    # extras_require={"plotting": ["matplotlib>=2.2.0", "jupyter"]},
    # setup_requires=["pytest-runner", "flake8"],
    # tests_require=["pytest"],
    entry_points={
        "console_scripts": ["mega_halo=mega.halo_core.main_halo:main",
                            "mega_graph=mega.graph_core.main_graph:main"]
    },
)
