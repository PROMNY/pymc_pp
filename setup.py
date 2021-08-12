from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='pymc_pp',
    version='0.1.0',
    description='Monte-Carlo simulations of Falicov-Kimball in python',
    long_description=readme,
    author='Paweł Promny',
    author_email='p.promny@gmail.com',
    url='https://github.com/PROMNY/pymc',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
