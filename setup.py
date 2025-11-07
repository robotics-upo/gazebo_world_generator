from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'gazebo_world_generator'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Config files
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        # Prompt templates
        (os.path.join('share', package_name, 'prompts', 'v1'), glob('prompts/v1/*.j2')),
        (os.path.join('share', package_name, 'prompts', 'v1'), glob('prompts/v1/*.json')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Miguel Escudero Jiménez',
    maintainer_email='mescjim@upo.es',
    description='LLM-based Gazebo world generator for robotic simulation environments',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'generate_world = gazebo_world_generator.gazebo_world_generator:main',
        ],
    },
)
