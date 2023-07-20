import launch
import launch_ros.actions

def generate_launch_description():
    environment = "00195"
    ld = launch.LaunchDescription([
        launch.actions.DeclareLaunchArgument(
	name='pose_publish_topic',
    	default_value='/spot/visual_pose_estimate'
        ),
        launch.actions.DeclareLaunchArgument(
    	name='camera_topic',
    	# default_value='/spot/rgb/image_rect_color'
    	default_value='/image_raw'
        ),
        launch.actions.DeclareLaunchArgument(
    	name='base_frame',
    	default_value='spot'
        ),
        launch.actions.DeclareLaunchArgument(
    	name='sensor_frame',
    	default_value='spot/rgb_front'
        ),
	launch.actions.DeclareLaunchArgument(
    	name='global_extractor_name',
    	default_value='netvlad'
        ),
	launch.actions.DeclareLaunchArgument(
    	name='local_extractor_name',
    	default_value='superpoint_aachen'
        ),
	launch.actions.DeclareLaunchArgument(
    	name='local_matcher_name',
    	default_value='NN-superpoint'
        ),
        launch.actions.DeclareLaunchArgument(
    	name='gallery_global_descriptor_path',
    	default_value=f'/opt/visual_robot_localization/src/visual_robot_localization/hloc_outputs/{environment}/global-feats-netvlad.h5'
        ),
        launch.actions.DeclareLaunchArgument(
    	name='gallery_local_descriptor_path',
    	default_value=f'/opt/visual_robot_localization/src/visual_robot_localization/hloc_outputs/{environment}/feats-superpoint-n4096-r1024.h5'
        ),
        launch.actions.DeclareLaunchArgument(
    	name='image_gallery_path',
    	default_value=f'/opt/visual_robot_localization/src/visual_robot_localization/hloc_datasets/{environment}/mapping'
        ),
        launch.actions.DeclareLaunchArgument(
    	name='gallery_sfm_path',
    	default_value=f'/opt/visual_robot_localization/src/visual_robot_localization/hloc_outputs/{environment}/reconstruction'
        ),
        launch.actions.DeclareLaunchArgument(
    	name='compensate_sensor_offset',
    	default_value='False'
        ),
        launch.actions.DeclareLaunchArgument(
    	name='localization_frequence',
    	default_value='3.0'
        ),
        # If the localization frequence is in ROS or wall time
        launch.actions.DeclareLaunchArgument(
    	name='use_sim_time',
    	default_value='False'
        ),
        launch.actions.DeclareLaunchArgument(
    	name='top_k_matches',
    	default_value='20'
        ),
        launch.actions.DeclareLaunchArgument(
    	name='ransac_thresh',
    	default_value='12'
        ),
        launch_ros.actions.Node(
            package='visual_robot_localization',
            executable='visual_localizer_node',
            output='screen',
            emulate_tty=True,
            parameters=[
                {
    		     'use_sim_time': launch.substitutions.LaunchConfiguration('use_sim_time')
                },
                {
                    'pose_publish_topic': launch.substitutions.LaunchConfiguration('pose_publish_topic')
                },
                {
                    'camera_topic': launch.substitutions.LaunchConfiguration('camera_topic')
                },
                {
                    'base_frame': launch.substitutions.LaunchConfiguration('base_frame')
                },
                {
                    'sensor_frame': launch.substitutions.LaunchConfiguration('sensor_frame')
                },
                {
                    'global_extractor_name': launch.substitutions.LaunchConfiguration('global_extractor_name')
                },
                {
                    'local_extractor_name': launch.substitutions.LaunchConfiguration('local_extractor_name')
                },
                {
                    'local_matcher_name': launch.substitutions.LaunchConfiguration('local_matcher_name')
                },
                {
                    'gallery_global_descriptor_path': launch.substitutions.LaunchConfiguration('gallery_global_descriptor_path')
                },
                {
                    'gallery_local_descriptor_path': launch.substitutions.LaunchConfiguration('gallery_local_descriptor_path')
                },
                {
                    'image_gallery_path': launch.substitutions.LaunchConfiguration('image_gallery_path')
                },
                {
                    'gallery_sfm_path': launch.substitutions.LaunchConfiguration('gallery_sfm_path')
                },
                {
                    'compensate_sensor_offset': launch.substitutions.LaunchConfiguration('compensate_sensor_offset')
                },
                {
                    'localization_frequence': launch.substitutions.LaunchConfiguration('localization_frequence')
                },
                {
                    'top_k_matches': launch.substitutions.LaunchConfiguration('top_k_matches')
                },
                {
                    'ransac_thresh': launch.substitutions.LaunchConfiguration('ransac_thresh')
                }
            ]
        )
    ])
    return ld


if __name__ == '__main__':
    generate_launch_description()
