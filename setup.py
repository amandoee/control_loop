from setuptools import setup

package_name = 'control_loop'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='amandoee',
    maintainer_email='amandoee@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'control_loop_keyboard = control_loop.control_keyboard:main',

            'control_loop = control_loop.control_loop_neural_conv_anticipate:main',

            'control_slam = control_loop.control_slam:main',

            'control_amcl = control_loop.control_amcl:main',

            'control_odom = control_loop.control_loop_neural_conv_anticipate_odom:main',

            'control_loop_pd = control_loop.control_loop_pd:main'
        ],
    },
)
