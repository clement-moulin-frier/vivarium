from setuptools import find_packages, setup

setup(
    name='vivarium',
    # extras_require=dict(tests=['pytest']),
    packages=find_packages(where="src"),
    package_dir={"": "src"}
)