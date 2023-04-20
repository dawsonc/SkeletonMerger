from setuptools import setup, find_packages
from pip._internal.req import parse_requirements

# parse_requirements() returns generator of pip.req.InstallRequirement objects
requirements = parse_requirements('requirements.txt', session='dummy')

# convert the generator to a list
install_requires = [str(req.requirement) for req in requirements]

setup(
    name='merger',
    version='0.1',
    description='SkeletonMerger',
    author='Charles Dawson',
    author_email='charles.dwsn@gmail.com',
    url='https://github.com/dawsonc/SkeletonMerger',
    packages=find_packages(),
    install_requires=install_requires,
)
