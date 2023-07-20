import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from cv_bridge import CvBridge
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration


from std_msgs.msg import ColorRGBA, Header, Bool
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose, Point, Quaternion, PoseArray, PoseWithCovariance, Vector3
from visualization_msgs.msg import Marker
from sensor_msgs.msg import PointCloud2, PointField
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

from pytransform3d import transformations as pt






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

        self.T_cam_base = pt.transform_from(
                np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0,
                                                               0.0]]),
                np.array([0, 0.0, 0.0]))

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
            visual_pose_estimate_msg, pose_msg = self._construct_visual_pose_msg(best_estimate, image_msg.header.stamp, true_delay)
            
            self.vloc_publisher.publish(visual_pose_estimate_msg)

            self.vloc_pose_publisher.publish(pose_msg)

            if self.visualize_estimates:
                self._estimate_visualizer(ret, image_msg.header.stamp, best_cluster_idx)


    def _construct_visual_pose_msg(self, best_estimate, timestamp, vloc_computation_delay):

        if best_estimate is not None:
            best_pose_msg = PoseWithCovariance(pose=Pose(position=np2point_msg(best_estimate['tvec']),
                                                        orientation=np2quat_msg(best_estimate['qvec'])),
                                                covariance = self.vloc_estimate_covariance)
            pnp_success = Bool(data=True)
        else:
            best_pose_msg = PoseWithCovariance()
            pnp_success = Bool(data=False)

        visual_pose_estimate_msg = VisualPoseEstimate(header=Header(frame_id='map', stamp=timestamp),
                                                    pnp_success = pnp_success,
                                                    computation_delay=vloc_computation_delay.to_msg(),
                                                    pose=best_pose_msg)
        return visual_pose_estimate_msg, best_pose_msg


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
        header = Header(frame_id='map', stamp=timestamp)
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

            if estimate['success'] and i == best_pose_idx:
                pose_msg = Pose(position=np2point_msg(estimate['tvec']), orientation=np2quat_msg(estimate['qvec']))
                poses.poses.append(pose_msg)

                static_t = TransformStamped()
                static_t.header.stamp = self.get_clock().now().to_msg()
                static_t.header.frame_id = 'map'
                static_t.child_frame_id = "colmap"
                pq = pt.pq_from_transform(self.T_cam_base)
                static_t.transform.translation.x = pq[0]
                static_t.transform.translation.y = pq[1]
                static_t.transform.translation.z = pq[2]
                static_t.transform.rotation.x = pq[4]
                static_t.transform.rotation.y = pq[5]
                static_t.transform.rotation.z = pq[6]
                static_t.transform.rotation.w = pq[3]

                self.tf_broadcaster.sendTransform(static_t)

                t = TransformStamped()
                t.header.stamp = self.get_clock().now().to_msg()
                t.header.frame_id = 'colmap'
                t.child_frame_id = "cam"

                

                p = estimate['tvec']
                q = estimate['qvec']
                pq = np.concatenate([p, q])

                tr = pt.transform_from_pq(pq)
                # tr = pt.invert_transform(tr)
                # tr = pt.invert_transform(tr)
                pq = pt.pq_from_transform(tr)

                # transform = self.transform_from_pq(pq)
                # #transform = np.linalg.inv(transform)
                # pq = self.pq_from_transform(transform)
          
                t.transform.translation.x = pq[0]
                t.transform.translation.y = pq[1]
                t.transform.translation.z = pq[2]
                
                # fixes translation
                # t.transform.translation.x = -pq[1]
                # t.transform.translation.y = -pq[2]
                # t.transform.translation.z = pq[0]

                # t.transform.translation.x = 1.0
                # t.transform.translation.y = 0.0
                # t.transform.translation.z = 0.0

                t.transform.rotation.x = pq[4]
                t.transform.rotation.y = pq[5]
                t.transform.rotation.z = pq[6]
                t.transform.rotation.w = pq[3]
                # t.transform.rotation.x = 0
                # t.transform.rotation.y = 0
                # t.transform.rotation.z = 0
                # t.transform.rotation.w = 1
                # Send the transformation
                self.tf_broadcaster.sendTransform(t)




        # self.place_recognition_publisher.publish(marker)
        self.pnp_estimate_publisher.publish(poses)
        self.pc_publisher.publish(self.pointcloud_msg)

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
    
    def qvec2rotmat(self, qvec) -> np.ndarray:
        return np.array([[
            1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
            2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
            2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]
        ],
            [
            2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
            1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
            2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]
        ],
            [
            2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
            2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
            1 - 2 * qvec[1]**2 - 2 * qvec[2]**2
        ]])

    def transform_from_pq(self, pq) -> np.ndarray:
        R = self.qvec2rotmat(pq[3:])
        t = np.array([pq[:3]]).T
        T = np.hstack((R, t))
        T = np.vstack((T, np.array([0, 0, 0, 1])))
        return T
    def pq_from_transform(self, T) -> np.ndarray:
        p: np.ndarray = T[:3, 3]
        q = self.rotmat2qvec(T[:3, :3])
        return np.concatenate((p, q))
    
    def rotmat2qvec(self, R) -> np.ndarray:
        Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
        K = np.array([[Rxx - Ryy - Rzz, 0, 0, 0], [
            Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0
        ], [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                    [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
        eigvals, eigvecs = np.linalg.eigh(K)
        qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
        if qvec[0] < 0:
            qvec *= -1
        return qvec


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
