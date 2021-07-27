import setuptools
from setuptools import setup

with open('requirements.txt', 'r') as f:
    install_requires = [s.replace('\n', '') for s in f.readlines()]

setup(name='outbreak',
      version='0.1',
      license='MIT',
      author='Frank Schlosser',
      author_email='frankfschlosser@gmail.com',
      description='Method to infer outbreak origins from individual trajectories.',
      long_description='',
      url='https://github.com/franksh/outbreak-detection',
      packages=setuptools.find_packages(),
      install_requires=install_requires,
      python_requires='>=3.7',
      dependency_links=[
      ],
      classifiers=['License :: OSI Approved :: MIT License',
                   'Programming Language :: Python :: 3.7',
                   'Programming Language :: Python :: 3.8',
                   ],
      tests_require=[],
      project_urls={
          'Source': 'https://github.com/franksh/outbreak-detection/',
      },
      include_package_data=True,
      zip_safe=False,
      )
