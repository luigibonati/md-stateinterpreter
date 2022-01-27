
"""
stateinterpreter
Interpretation of metastable states from MD simulations
"""
import sys
from setuptools import setup, find_packages, Extension
import versioneer
import numpy

os_name = sys.platform
compile_args = ["-O3", "-ffast-math", "-march=native", "-fopenmp" ]
libraries = ["m"]
link_args = ['-fopenmp']
if os_name.startswith('darwin'):
    #clang compilation
    compile_args.insert(-1, "-Xpreprocessor")
    libraries.append("omp")
    link_args.insert(-1, "-Xpreprocessor")

__cython__ = False   # command line option, try-import, ...
ext = '.pyx' if __cython__ else '.c'

short_description = "Interpretation of metastable states from MD simulations".split("\n")[0]

# from https://github.com/pytest-dev/pytest-runner#conditional-requirement
needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
pytest_runner = ['pytest-runner'] if needs_pytest else []

try:
    with open("README.md", "r") as handle:
        long_description = handle.read()
except:
    long_description = None

ext_modules=[
    Extension("stateinterpreter.utils._compiled_numerics",
            ["stateinterpreter/utils/_compiled_numerics.pyx"],
            libraries=libraries,
            include_dirs=[numpy.get_include()],
            extra_compile_args = compile_args,
            extra_link_args= link_args
    ) 
]

if __cython__:
    from Cython.Build import cythonize
    ext_modules = cythonize(ext_modules)

setup(
    # Self-descriptive entries which should always be present
    name='stateinterpreter',
    author='Luigi Bonati <luigi.bonati@iit.it>, Pietro Novelli <pietro.novelli.iit>"',
    description=short_description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    license='MIT',

    # Which Python importable modules should be included when your package is installed
    # Handled automatically by setuptools. Use 'exclude' to prevent some specific
    # subpackage(s) from being added, if needed
    packages=find_packages(),

    # Optional include package data to ship with your package
    # Customize MANIFEST.in if the general case does not suit your needs
    # Comment out this line to prevent the files from being packaged with your software
    include_package_data=True,

    # Allows `setup.py test` to work correctly with pytest
    setup_requires=[] + pytest_runner,
    ext_modules = ext_modules,
    zip_safe = False,

    # Additional entries you may want simply uncomment the lines you want and fill in the data
    # url='http://www.my_package.com',  # Website
    # install_requires=[],              # Required packages, pulls from pip if needed; do not use for Conda deployment
    # platforms=['Linux',
    #            'Mac OS-X',
    #            'Unix',
    #            'Windows'],            # Valid platforms your code works on, adjust to your flavor
    # python_requires=">=3.5",          # Python version restrictions

    # Manual control if final package is compressible or not, set False to prevent the .egg from being made
    # zip_safe=False,
)