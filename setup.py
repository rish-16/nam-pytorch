from setuptools import setup, find_packages

with open('README.md') as readme_file:
    README = readme_file.read()

setup(
  name = 'nam_pytorch',
  packages = find_packages(exclude=[]),
  version = '0.0.1',
  license='MIT',
  description = 'Neural Additive Models (NAM) - Pytorch',
  long_description_content_type="text/markdown",
  long_description=README,
  author = 'Rishabh Anand',
  author_email = 'mail.rishabh.anand@gmail.com',
  url = 'https://github.com/rish-16/nam-pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'nam',
    'neural additive models',
    'generalized additive models'
  ],
  install_requires=[
    'torch>=1.6'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)