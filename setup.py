from setuptools import setup, find_packages

__author__ = 'Haoyan Huo'
__maintainer__ = 'Haoyan Huo'
__email__ = 'haoyan.huo@lbl.gov'

if __name__ == '__main__':
    setup(
        name='s4',
        author='Haoyan Huo',
        author_email='hhaoyann@gmail.com',
        url='https://github.com/CederGroupHub/s4',
        description='Synthesis Science mining from Solid-state Synthesis dataset',
        classifiers=[
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3 :: Only',
            'Development Status :: 1 - Planning',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: MIT License',
            'Operating System :: OS Independent',
            'Topic :: Scientific/Engineering :: Information Analysis',
            'Topic :: Scientific/Engineering :: Physics',
            'Topic :: Scientific/Engineering :: Chemistry',
            'Topic :: Software Development :: Libraries :: Python Modules'
        ],
        python_requires=">=3.6",
        license='MIT',
        include_package_data=True,
        version='0.1.0',
        packages=find_packages(),
        install_requires=open('requirements.txt').readlines()
    )
