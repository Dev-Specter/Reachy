from setuptools import find_packages, setup

package_name = 'reachy_rl_control'

setup(
    name=package_name,
    version='0.0.1',

    packages=find_packages(),

    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name],
        ),
        (
            'share/' + package_name,
            ['package.xml'],
        ),
       (
            'share/' + package_name + '/models',
            ['models/human_contact.sdf'],
       ),
    ],

    install_requires=['setuptools'],

    zip_safe=True,

    maintainer='abhinav',
    maintainer_email='abhinav@example.com',

    description='Reachy Gazebo reinforcement learning environment',

    license='MIT',

    entry_points={
        'console_scripts': [
            'train_nav = reachy_rl_control.train_nav:main',
            'evaluate_nav = reachy_rl_control.evaluate_nav:main',
            'make_eval_set = reachy_rl_control.make_eval_set:main',
            'probe_free_space = reachy_rl_control.probe_free_space:main',
        ],
    },
)
