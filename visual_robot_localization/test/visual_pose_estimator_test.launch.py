import launch
import launch_ros.actions

import pytest
import unittest
import launch_testing
import rclpy
import sys
import os

@pytest.mark.rostest
def generate_test_description():

    path_to_test = os.path.dirname(__file__)
    environment = "DLAB_Test"
    
    test_node = launch_ros.actions.Node(
            executable=sys.executable,
            arguments=[ path_to_test + '/visual_localizer_node_test_node.py'],
            output='screen',
            emulate_tty=True,
            parameters=[
            {
            	'image_publish_topic': '/spot/rgb_front/image'
            },
            {
            	'visual_pose_estimate_subscription_topic': '/spot/visual_pose_estimate'
            },
            {
            	'test_image_path':  f'/opt/visual_robot_localization/src/visual_robot_localization/hloc_datasets/{environment}/mapping/hetlf_1596980888.jpg'
    	    }
    	    ]
            )
            
    localizer_node = launch_ros.actions.Node(
            package='visual_robot_localization',
            executable='visual_localizer_node',
            output='screen',
            emulate_tty=True,
            parameters=[
                {    # If the localization frequence is in ROS or wall time
    		     'use_sim_time': False
                },
                {
                    'pose_publish_topic': '/spot/visual_pose_estimate'
                },
                {
                    'camera_topic': '/spot/rgb_front/image'
                },
                {
                    'global_extractor_name': 'netvlad'
                },
                {
                    'local_extractor_name': 'superpoint_aachen'
                },
                {
                    'local_matcher_name': 'NN-superpoint'
                },
                {
                    'gallery_global_descriptor_path': f'/opt/visual_robot_localization/src/visual_robot_localization/hloc_outputs/{environment}/global-feats-netvlad.h5' # '/example_dir/outputs/dir+superpoint_aachen+superglue/global-feats-dir.h5' #
                },
                {
                    'gallery_local_descriptor_path': f'/opt/visual_robot_localization/src/visual_robot_localization/hloc_outputs/{environment}/feats-superpoint-n4096-r1024.h5' # '/example_dir/outputs/dir+superpoint_aachen+superglue/feats-superpoint-n4096-r1024.h5' 
                },
                {
                    'image_gallery_path':  f'/opt/visual_robot_localization/src/visual_robot_localization/hloc_datasets/{environment}/mapping'
                },
                {
                    'gallery_sfm_path': f'/opt/visual_robot_localization/src/visual_robot_localization/hloc_outputs/{environment}/reconstruction' # '/example_dir/outputs/dir+superpoint_aachen+superglue/sfm_dir+superpoint_aachen+superglue' 
                },
                {
                    'compensate_sensor_offset': False
                },
                {
                    'localization_frequence': 2.0
                },
                {
                    'top_k_matches': 5
                },
                {
                    'ransac_thresh': 12
                }
            ]
        )
            
    ld = launch.LaunchDescription([
        localizer_node,
        test_node,
        launch.actions.RegisterEventHandler(
            launch.event_handlers.OnProcessExit(
                target_action=test_node,
                on_exit=[
                    launch.actions.LogInfo(msg='Test completed'),
                    launch.actions.EmitEvent(event=launch.events.Shutdown(
                        reason='Test completed\n'))
                ]
            )
        ),
        launch_testing.actions.ReadyToTest(),
    ])
    return ld

class BasicUsageTest(unittest.TestCase):
        
    def test_node_start(self, proc_output):
        rclpy.init()
        msg = 'Got Result!'
        proc_output.assertWaitFor(msg, process=None, stream='stdout', timeout=120)
        rclpy.shutdown()

