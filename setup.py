from setuptools import setup, find_packages
from setuptools.dist import Distribution
import os
import sys
from glob import glob

package_name = 'gazebo_world_generator'


class _Distribution(Distribution):
    """Keep setup.cfg's ROS script directory out of pip wheels.

    colcon needs executables in lib/<package> for `ros2 run`; pip and pipx
    need them in bin/. colcon never builds wheels, so wheel builds drop it.
    """

    def parse_config_files(self, *args, **kwargs):
        super().parse_config_files(*args, **kwargs)
        if {'bdist_wheel', 'editable_wheel'} & set(sys.argv):
            self.command_options.get('install', {}).pop('install_scripts', None)
            self.command_options.get('develop', {}).pop('script_dir', None)


setup(
    distclass=_Distribution,
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
    install_requires=[
        'setuptools', 'openai>=1.0.0', 'jsonschema>=4.0.0',
        'json5>=0.9.0', 'requests>=2.28.0', 'PyYAML>=6.0',
        'pydantic>=2.0.0', 'pydantic-settings>=2.0.0',
        'Jinja2>=3.1.0',
    ],
    python_requires='>=3.10',
    zip_safe=True,
    maintainer='Miguel Escudero Jiménez',
    maintainer_email='mescjim@upo.es',
    description='LLM-based Gazebo world generator for robotic simulation environments',
    license='MIT',
    entry_points={
        'console_scripts': [
            'generate_world = gazebo_world_generator.gazebo_world_generator:main',
        ],
    },
)
