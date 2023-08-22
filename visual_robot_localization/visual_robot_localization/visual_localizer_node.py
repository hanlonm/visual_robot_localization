import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from cv_bridge import CvBridge
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration


from std_msgs.msg import ColorRGBA, Header, Bool
from std_srvs.srv import SetBool, Empty
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose, Point, Quaternion, PoseArray, PoseWithCovariance, Vector3
from visualization_msgs.msg import Marker
from sensor_msgs.msg import PointCloud2, PointField
from tf2_ros import TransformBroadcaster, TransformListener, StaticTransformBroadcaster
from tf2_ros.buffer import Buffer
import tf2_geometry_msgs

from geometry_msgs.msg import TransformStamped

import pytransform3d.transformations as pt


from visual_localization_interfaces.msg import VisualPoseEstimate
from visual_localization_interfaces.msg import VisualLocalizerStatus


import cv2
from copy import deepcopy
import threading
import numpy as np
from matplotlib import cm
import time

from visual_robot_localization.visual_6dof_localize import VisualPoseEstimator
from visual_robot_localization.coordinate_transforms import SensorOffsetCompensator

class VisualLocalizer(Node):
    def __init__(self):
        super().__init__('visual_localization_node')

        self.declare_parameter("pose_publish_topic", "/carla/ego_vehicle/visual_pose_estimate")
        pose_publish_topic = self.get_parameter('pose_publish_topic').get_parameter_value().string_value

        self.declare_parameter("camera_topic", "/carla/ego_vehicle/rgb_front/image")
        camera_topic_name = self.get_parameter('camera_topic').get_parameter_value().string_value

        self.declare_parameter("global_extractor_name", "netvlad")
        global_extractor_name = self.get_parameter('global_extractor_name').get_parameter_value().string_value

        self.declare_parameter("local_extractor_name", "superpoint_aachen")
        local_extractor_name = self.get_parameter('local_extractor_name').get_parameter_value().string_value

        self.declare_parameter("local_matcher_name", "superglue")
        local_matcher_name = self.get_parameter('local_matcher_name').get_parameter_value().string_value

        self.declare_parameter("gallery_global_descriptor_path", "/image-gallery/example_dir/outputs/netvlad+superpoint_aachen+superglue/global-feats-netvlad.h5")
        gallery_global_descriptor_path = self.get_parameter('gallery_global_descriptor_path').get_parameter_value().string_value

        self.declare_parameter("gallery_local_descriptor_path", "/image-gallery/example_dir/outputs/netvlad+superpoint_aachen+superglue/feats-superpoint-n4096-r1024.h5")
        gallery_local_descriptor_path = self.get_parameter('gallery_local_descriptor_path').get_parameter_value().string_value

        self.declare_parameter("image_gallery_path", "/image-gallery/example_dir/")
        image_gallery_path = self.get_parameter('image_gallery_path').get_parameter_value().string_value

        self.declare_parameter("gallery_sfm_path", "/image-gallery/example_dir/outputs/sfm_netvlad+superpoint_aachen+superglue/")
        gallery_sfm_path = self.get_parameter('gallery_sfm_path').get_parameter_value().string_value

        self.declare_parameter("cam_string", "PINHOLE 1280 720 609.5238037109375 610.1694946289062 640 360")
        cam_string = self.get_parameter('cam_string').get_parameter_value().string_value

        self.declare_parameter("localization_frequence", 2.0)
        self.pr_freq = self.get_parameter('localization_frequence').get_parameter_value().double_value

        self.declare_parameter("top_k_matches", 4)
        self.top_k = self.get_parameter('top_k_matches').get_parameter_value().integer_value

        self.declare_parameter("compensate_sensor_offset", True)
        self.compensate_sensor_offset = self.get_parameter('compensate_sensor_offset').get_parameter_value().bool_value

        self.declare_parameter("base_frame", "ego_vehicle")
        base_frame = self.get_parameter('base_frame').get_parameter_value().string_value

        self.declare_parameter("sensor_frame", "ego_vehicle/rgb_front")
        sensor_frame = self.get_parameter('sensor_frame').get_parameter_value().string_value

        self.declare_parameter("align_camera_frame", True)
        align_camera_frame = self.get_parameter('align_camera_frame').get_parameter_value().bool_value

        self.declare_parameter("ransac_thresh", 12)
        ransac_thresh = self.get_parameter('ransac_thresh').get_parameter_value().integer_value

        self.declare_parameter("visualize_estimates", True)
        self.visualize_estimates = self.get_parameter('visualize_estimates').get_parameter_value().bool_value

        self.vloc_publisher = self.create_publisher(
            VisualPoseEstimate,
            pose_publish_topic,
            10)
        self.vloc_pose_publisher = self.create_publisher(
            PoseWithCovariance,
            pose_publish_topic + "_pose",
            10)
        
        self.pc_publisher = self.create_publisher(PointCloud2, 'map_pointcloud', 10)

        # Initialize the transform broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)


        # Transforms
        self.T_rotated_colmap = pt.transform_from(
                        np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]),
                        np.array([0, 0.0, 0.0]))
        self.T_colmap_rotated = pt.invert_transform(self.T_rotated_colmap)
        self.tf_rotated_colmap = self.transform_to_tf(T_frameId_childFrameId=self.T_colmap_rotated, frame_id="colmap", child_frame_id="rotated")

        self.T_colmap_cam = None
        self.T_map_odom = None
        self.T_map_body_est = None

        # Visualization publishers
        if self.visualize_estimates:
            self.place_recognition_publisher = self.create_publisher(Marker, '/place_recognition_visualization', 10)
            self.pnp_estimate_publisher = self.create_publisher(PoseArray, '/visual_pose_estimate_visualization', 10)
            self.colormap = cm.get_cmap('Accent')

        self.subscription = self.create_subscription(
            Image,
            camera_topic_name,
            self.camera_subscriber_callback,
            10)

        self.timer = self.create_timer( 1/self.pr_freq , self.computation_callback, callback_group = ReentrantCallbackGroup())
        self.tf_timer = self.create_timer( 1/10 , self.publish_transforms, callback_group = ReentrantCallbackGroup())

        self.update_loc = False

        self.lock = threading.Lock()
        self.latest_image = None
        self.cv_bridge = CvBridge()
        
        self.pose_estimator = VisualPoseEstimator(global_extractor_name,
                                                    local_extractor_name,
                                                    local_matcher_name,
                                                    image_gallery_path,
                                                    gallery_global_descriptor_path,
                                                    gallery_local_descriptor_path,
                                                    gallery_sfm_path,
                                                    cam_string,
                                                    ransac_thresh)
        self.reconstruction = self.pose_estimator.reconstruction
        self.landmark_list = list(self.reconstruction.points3D.keys())
        self.pcd_points = np.zeros((len(self.landmark_list), 3))
         
        for idx, landmark in enumerate(self.landmark_list):
            self.pcd_points[idx] = np.array(
                self.reconstruction.points3D[landmark].xyz)
        self.pointcloud_msg = self.point_cloud(points=self.pcd_points, parent_frame="map")
        self.pc_publisher.publish(self.pointcloud_msg)

        if self.compensate_sensor_offset:
            self.get_logger().info('Constructing sensor offset compensator...')
            self.sensor_offset_compensator = SensorOffsetCompensator(base_frame, sensor_frame, align_camera_frame)

            if not hasattr(self.sensor_offset_compensator.tvec, '__len__') and not hasattr(self.sensor_offset_compensator.qvec, '__len__'):
                self.status_publisher = self.create_publisher(VisualLocalizerStatus,
                                                                        "/visual_localizer/status",
                                                                        10)
                msg = VisualLocalizerStatus()
                msg.status = 4
                self.status_publisher.publish(msg)

        loc_var = 0.1
        loc_cov = 0.0
        or_var = 0.1
        or_cov = 0.0
        self.vloc_estimate_covariance =    [loc_var,    loc_cov,    loc_cov,    0.0,    0.0,    0.0,
                                            loc_cov,    loc_var,    loc_cov,    0.0,    0.0,    0.0,
                                            loc_cov,    loc_cov,    loc_var,    0.0,    0.0,    0.0,
                                            0.0,    0.0,    0.0,    or_var,    or_cov,    or_cov,
                                            0.0,    0.0,    0.0,    or_cov,    or_var,    or_cov,
                                            0.0,    0.0,    0.0,    or_cov,    or_cov,    or_var]
        self.vloc_estimate_covariance = np.array(self.vloc_estimate_covariance)

        self.update_localization_srv = self.create_service(SetBool, 'updtate_localization', self.update_localization_callback)
        self.tag_localization_srv = self.create_service(Empty, 'tag_localization', self.tag_localization_callback)

    def update_localization_callback(self, request: SetBool.Request, response: SetBool.Response):
        self.update_loc = request.data
        self.get_logger().info(f'Localization publishing status: {self.update_loc}')
        response.success = True
        response.message = f'Localization publishing status: {self.update_loc}'
        return response
    
    def tag_localization_callback(self, request: Empty.Request, response: Empty.Response):
        try:
            tf_map_loctag: TransformStamped = self.tf_buffer.lookup_transform(
                target_frame="map",
                source_frame="loc_tag",
                time=rclpy.time.Time())
            tf_tag_odom: TransformStamped = self.tf_buffer.lookup_transform(
                target_frame="tag_1",
                source_frame="odom",
                time=rclpy.time.Time())
            
            T_map_loctag = self.tf_to_transform(tf_map_loctag)
            T_tag_odom = self.tf_to_transform(tf_tag_odom)

            T_map_odom = T_map_loctag @ T_tag_odom
            pub_time = tf_tag_odom.header.stamp
            tf_odom_map = self.transform_to_tf(T_frameId_childFrameId=T_map_odom, frame_id="map", child_frame_id="odom")
            self.tf_static_broadcaster.sendTransform(tf_odom_map)

        except Exception as ex:
            self.get_logger().info(
                f'Could not transform: {ex}')

        return response

    def camera_subscriber_callback(self, image_msg):
        '''
        Use the camera subscriber callback only for updating the image data
        '''
        with self.lock:
            self.latest_image = image_msg

    def computation_callback(self):
        '''
        Perform the heavy computation inside the timer callback
        '''
        with self.lock:
            computation_start_time = time.time_ns()
            image_msg = deepcopy(self.latest_image)

        if image_msg is not None:
    
            # Convert your ROS Image message to OpenCV2
            cv2_img = self.cv_bridge.imgmsg_to_cv2(image_msg, "bgr8")
            cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

            ret = self.pose_estimator.estimate_pose(cv2_img, self.top_k)

            if self.compensate_sensor_offset:
                # Compute the pose of the vehicle the camera is attached to
                for pose in ret['place_recognition']:
                    pose['tvec'], pose['qvec'] = self.sensor_offset_compensator.remove_offset_from_array(pose['tvec'], pose['qvec'])

                for pose in ret['pnp_estimates']:
                    if pose['success']:
                        pose['tvec'], pose['qvec'] = self.sensor_offset_compensator.remove_offset_from_array(pose['tvec'], pose['qvec'])
            
            best_estimate, best_cluster_idx = self.choose_best_estimate(ret)

            true_delay = Duration(nanoseconds = time.time_ns()-computation_start_time)
            visual_pose_estimate_msg = self._construct_visual_pose_msg(best_estimate, image_msg.header.stamp, true_delay)
            
            self.vloc_publisher.publish(visual_pose_estimate_msg)
            if best_estimate is not None:
                p = best_estimate['tvec']
                q = best_estimate['qvec']
                pq = np.concatenate([p, q])

                # print("start")
                # print(pt.transform_from_pq(pq))
                # print(pt.invert_transform(pt.transform_from_pq(pq)))

                # This is where I have no idea what transformations are happening
                T_rotated_cam = pt.transform_from_pq(pq)
                self.T_colmap_cam = self.T_colmap_rotated @ T_rotated_cam

                try:
                    tf_map_colmap: TransformStamped = self.tf_buffer.lookup_transform(
                        target_frame="map",
                        source_frame="colmap",
                        time=rclpy.time.Time())
                    
                    # tf_hand_odom: TransformStamped = self.tf_buffer.lookup_transform(
                    #     target_frame="hand",
                    #     source_frame="odom",
                    #     time=rclpy.time.Time())
                    tf_handimg_odom: TransformStamped = self.tf_buffer.lookup_transform(
                        target_frame="hand_color_image_sensor",
                        source_frame="odom",
                        time=rclpy.time.Time())
                    tf_handimg_body: TransformStamped = self.tf_buffer.lookup_transform(
                        target_frame="hand_color_image_sensor",
                        source_frame="body",
                        time=rclpy.time.Time())
                    tf_loccam_locimg: TransformStamped = self.tf_buffer.lookup_transform(
                        target_frame="loc_cam",
                        source_frame="loc_img",
                        time=rclpy.time.Time())

                    T_handimg_body = self.tf_to_transform(tf_handimg_body)
                    T_loccam_locimg = self.tf_to_transform(tf_loccam_locimg)
                    T_map_colmap = self.tf_to_transform(tf_map_colmap)
                    T_map_cam = T_map_colmap @ self.T_colmap_cam
                    T_map_locimg = T_map_cam @ T_loccam_locimg

                    # T_hand_odom = self.tf_to_transform(tf_hand_odom)
                    T_handimg_odom = self.tf_to_transform(tf_handimg_odom)

                    # self.T_map_odom =  T_map_cam @ T_hand_odom
                    self.T_map_odom =  T_map_locimg @ T_handimg_odom
                    self.T_map_body_est = T_map_locimg @ T_handimg_body
                                       
                    
                except Exception as ex:
                    self.get_logger().info(
                        f'Could not transform: {ex}')
                    return

            if self.visualize_estimates:
                self._estimate_visualizer(ret, image_msg.header.stamp, best_cluster_idx)

    def publish_transforms(self):
        self.tf_broadcaster.sendTransform(self.tf_rotated_colmap)
        if self.T_colmap_cam is not None and self.T_map_odom is not None:
            tf_cam_colmap = self.transform_to_tf(T_frameId_childFrameId=self.T_colmap_cam, frame_id="colmap", child_frame_id="loc_cam")
            self.tf_broadcaster.sendTransform(tf_cam_colmap)

            tf_map_body = self.transform_to_tf(T_frameId_childFrameId=self.T_map_body_est,frame_id="map", child_frame_id="body_est")
            self.tf_broadcaster.sendTransform(tf_map_body)
            
            if self.update_loc:
                try:
                    tf_hand_odom: TransformStamped = self.tf_buffer.lookup_transform(
                            target_frame="hand",
                            source_frame="odom",
                            time=rclpy.time.Time())
                    pub_time = tf_hand_odom.header.stamp
                    tf_odom_map = self.transform_to_tf(T_frameId_childFrameId=self.T_map_odom, frame_id="map", child_frame_id="odom", time=pub_time)
                    self.tf_static_broadcaster.sendTransform(tf_odom_map)

                except Exception as ex:
                    self.get_logger().info(
                        f'Could not transform: {ex}')
                    return

    def tf_to_transform(self, tf: TransformStamped) -> np.ndarray:
        pq = np.array([tf.transform.translation.x,
                       tf.transform.translation.y,
                       tf.transform.translation.z,
                       tf.transform.rotation.w,
                       tf.transform.rotation.x,
                       tf.transform.rotation.y,
                       tf.transform.rotation.z])
        
        return pt.transform_from_pq(pq)
    
    def transform_to_tf(self, T_frameId_childFrameId: np.ndarray, frame_id: str, child_frame_id: str, time=None) -> TransformStamped:
        
        pq = pt.pq_from_transform(T_frameId_childFrameId)

        tf = TransformStamped()
        if time == None:
            tf.header.stamp = self.get_clock().now().to_msg()
        else:
            tf.header.stamp = time
        tf.header.frame_id = frame_id
        tf.child_frame_id = child_frame_id
        tf.transform.translation.x = pq[0]
        tf.transform.translation.y = pq[1]
        tf.transform.translation.z = pq[2]
    
        tf.transform.rotation.x = pq[4]
        tf.transform.rotation.y = pq[5]
        tf.transform.rotation.z = pq[6]
        tf.transform.rotation.w = pq[3]
        return tf


    def _construct_visual_pose_msg(self, best_estimate, timestamp, vloc_computation_delay):

        if best_estimate is not None:
            best_pose_msg = PoseWithCovariance(pose=Pose(position=np2point_msg(best_estimate['tvec']),
                                                        orientation=np2quat_msg(best_estimate['qvec'])),
                                                covariance = self.vloc_estimate_covariance)
            pnp_success = Bool(data=True)
        else:
            best_pose_msg = PoseWithCovariance()
            pnp_success = Bool(data=False)

        visual_pose_estimate_msg = VisualPoseEstimate(header=Header(frame_id='colmap', stamp=timestamp),
                                                    pnp_success = pnp_success,
                                                    computation_delay=vloc_computation_delay.to_msg(),
                                                    pose=best_pose_msg)
        return visual_pose_estimate_msg


    def choose_best_estimate(self, visual_pose_estimates):
        # Choose the pose estimate based on the number of ransac inliers
        best_inliers = 0
        best_estimate = None
        best_idx = None
        for i, estimate in enumerate(visual_pose_estimates['pnp_estimates']):
            if estimate['success']:
                if estimate['num_inliers'] > best_inliers:
                    best_inliers = estimate['num_inliers']
                    best_estimate = estimate
                    best_idx = i

        return best_estimate, best_idx


    def _estimate_visualizer(self, ret, timestamp, best_pose_idx):
        # Place recognition & PnP localization visualizations
        header = Header(frame_id='loc_cam', stamp=timestamp)
        # marker = Marker(header=header, scale=Vector3(x=1.0,y=1.0,z=1.0), type=8, action=0, color=ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0))
        poses = PoseArray(header=header, poses=[])
        for i, estimate in enumerate(ret['pnp_estimates']):
            if i == best_pose_idx:
                color = self.colormap(0)
            else:
                color = self.colormap(i+1)
            color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=color[3])

            # place_recognition_idxs = estimate['place_recognition_idx']
            # for idx in place_recognition_idxs:
            #     marker.colors.append(color)

            #     place_reg_position = np2point_msg(ret['place_recognition'][idx]['tvec'])
            #     marker.points.append(place_reg_position)

            if estimate['success']:
                # pose_msg = Pose(position=np2point_msg(estimate['tvec']), orientation=np2quat_msg(estimate['qvec']))
                pose_msg = Pose(position=np2point_msg([0.0,0.0,0.0]), orientation=np2quat_msg([1.0,0.0,0.0,0.0]))
                poses.poses.append(pose_msg)

        # self.place_recognition_publisher.publish(marker)
        self.pnp_estimate_publisher.publish(poses)
        # self.pc_publisher.publish(self.pointcloud_msg)

    def point_cloud(self, points, parent_frame):
        """ Creates a point cloud message.
        Args:
            points: Nx3 array of xyz positions.
            parent_frame: frame in which the point cloud is defined
        Returns:
            sensor_msgs/PointCloud2 message
        Code source:
            https://gist.github.com/pgorczak/5c717baa44479fa064eb8d33ea4587e0
        """
        ros_dtype = PointField.FLOAT32
        dtype = np.float32
        itemsize = np.dtype(dtype).itemsize  # A 32-bit float takes 4 bytes.

        data = points.astype(dtype).tobytes()
        fields = [PointField(
            name=n, offset=i * itemsize, datatype=ros_dtype, count=1)
            for i, n in enumerate('xyz')]
        header = Header(frame_id=parent_frame)

        return PointCloud2(
            header=header,
            height=1,
            width=points.shape[0],
            is_dense=False,
            is_bigendian=False,
            fields=fields,
            # Every point consists of three float32s.
            point_step=(itemsize * 3),
            row_step=(itemsize * 3 * points.shape[0]),
            data=data
        )


def np2point_msg(np_point):
    msg_point = Point(x=np_point[0], 
                    y=np_point[1], 
                    z=np_point[2])
    return msg_point

def np2quat_msg(np_quat):
    msg_quat = Quaternion(w=np_quat[0],
                        x=np_quat[1],
                        y=np_quat[2],
                        z=np_quat[3])
    return msg_quat

def main(args=None):

    rclpy.init(args=args)
    localizer = VisualLocalizer()
    try:
        executor = SingleThreadedExecutor()
        executor.add_node(localizer)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        localizer.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
