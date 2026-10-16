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
    ],

    install_requires=['setuptools'],

    zip_safe=True,

    maintainer='abhinav',
    maintainer_email='abhinav@example.com',

    description='Reachy Gazebo reinforcement learning environment',

    license='MIT',

    entry_points={
        'console_scripts': [
            'reachy_gazebo_interface = '
            'reachy_rl_control.reachy_gazebo_interface:main',

            'test_reset = '
            'reachy_rl_control.test_reset:main',

            'test_environment = '
            'reachy_rl_control.test_environment:main',

	   'test_trajectory = '
           'reachy_rl_control.test_trajectory:main',

	   'test_rl_environment = '
           'reachy_rl_control.test_rl_environment:main',
          
           'check_rl_environment = '
           'reachy_rl_control.check_rl_environment:main',

           'train_ppo = '
           'reachy_rl_control.train_ppo:main',

           'evaluate_ppo = '
           'reachy_rl_control.evaluate_ppo:main',

           'evaluate_ppo_multi = ''reachy_rl_control.evaluate_ppo_multi:main',
           'train_sac = reachy_rl_control.train_sac:main',
           'evaluate_sac_multi = reachy_rl_control.evaluate_sac_multi:main',
           'train_mbpo = reachy_rl_control.train_mbpo:main',
           'evaluate_mbpo_multi = reachy_rl_control.evaluate_mbpo_multi:main',
        ],
    },
)
