# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import io
import os
import setuptools

# https://packaging.python.org/guides/making-a-pypi-friendly-readme/
this_directory = os.path.abspath(os.path.dirname(__file__))
with io.open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
  long_description = f.read()

INSTALL_REQUIRES = ['jax-md']

setuptools.setup(
    name='vivarium',
    version='0.0.1',
    license='MIT',
    author='Clément Moulin-Frier',
    author_email='clement.moulinfrier@gmail.com',
    install_requires=INSTALL_REQUIRES,
    #url='https://github.com/google/jax-md',
    packages=setuptools.find_packages(),
    #download_url = "https://pypi.org/project/jax-md/",
    #project_urls={
    #    "Source Code": "https://github.com/google/jax-md",
    #    "Documentation": "https://arxiv.org/abs/1912.04232",
    #    "Bug Tracker": "https://github.com/google/jax-md/issues",
    #},
    #long_description=long_description,
    #long_description_content_type='text/markdown',
    #description='Differentiable, Hardware Accelerated, Molecular Dynamics',
    #python_requires='>=2.7',
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        # 'License :: OSI Approved :: Apache Software License',
        'Operating System :: MacOS',
        'Operating System :: POSIX :: Linux',
        'Topic :: Scientific/Engineering',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
    ])