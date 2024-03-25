#include <boost/filesystem.hpp>

#include <ros/ros.h>

#include <pcl/io/io.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl_ros/point_cloud.h>

#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <stereo_msgs/DisparityImage.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/Bool.h>

#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/time_synchronizer.h>

#include <cv_bridge/cv_bridge.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include <Eigen/Dense>

#include <string>
#include <time.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <utility>

#include <cuda_runtime.h>

#include <libsgm.h>

using namespace std;
using namespace sensor_msgs;
using namespace std_msgs;
using namespace stereo_msgs;
using namespace message_filters::sync_policies;


class stereoROS
{
  private:
    ros::NodeHandle nh_;
    boost::shared_ptr<image_transport::ImageTransport> it_;
    
    // Subscriptions
    image_transport::SubscriberFilter sub_l_image_, sub_r_image_;
    message_filters::Subscriber<CameraInfo> sub_l_info_, sub_r_info_;

    typedef ExactTime<Image, Image> ExactPolicy;
    typedef ApproximateTime<Image, Image> ApproximatePolicy;
    typedef message_filters::Synchronizer<ExactPolicy> ExactSync;
    typedef message_filters::Synchronizer<ApproximatePolicy> ApproximateSync;
    boost::shared_ptr<ExactSync> exact_sync_;
    boost::shared_ptr<ApproximateSync> approximate_sync_;

    ros::Publisher pub_disp_, pub_disp_color_;
    ros::Subscriber saver_sub_, stop_saver_sub_;

  public:
    struct device_buffer
    {
      device_buffer() : data(nullptr), size(0) {}
      device_buffer(size_t count) : device_buffer() { allocate(count); }
      ~device_buffer() { cudaFree(data); }

      void allocate(size_t count) { cudaMalloc(&data, count); size = count; }
      void upload(const void* h_data) { cudaMemcpy(data, h_data, size, cudaMemcpyHostToDevice); }
      void download(void* h_data) { cudaMemcpy(h_data, data, size, cudaMemcpyDeviceToHost); }

      void* data;
      size_t size;
    };

    int queue_size = 1;
    bool approx = false;
    int disp_size = 64;
    int p1 = 10;
    int p2 = 120;
    std::string save_dir = "/home/iot01/catkin_ws/src/stereo_ros/output/";
    std::string left_dir, right_dir, disp_dir, img_suffix;
    bool view_input = false, view_output = false;
    bool save_or_not = false;
    cv::Mat l_image, r_image;
    cv::Mat I1, I2;
    cv::Mat disparity;
    cv::Mat disparity_color;
    double scale = 0.5;
    int offset_x = 0;
    int offset_y = 0;
    int crop_h = 576;
    int crop_w = 1024;
    float uniqueness = 0.95;
    int num_paths = 8;
    int min_disp = 0;
    int LR_max_diff = 1;
    int cen = 0;
    int height, width;
    int src_depth = 8;
    int dst_depth = 8;
    int invalid_disp;

	  device_buffer d_I1, d_I2, d_disparity;

    stereoROS(ros::NodeHandle* nodehandle);
    void setParam();
    void imageCb(const ImageConstPtr& l_image_msg, const ImageConstPtr& r_image_msg);
    void imageWithInfoCb(const ImageConstPtr& l_image_msg, const CameraInfoConstPtr& l_info_msg,
                const ImageConstPtr& r_image_msg, const CameraInfoConstPtr& r_info_msg);
    void makeOutputFolder(std::string folderName);
    void saverCallback(const Bool::ConstPtr& msg);
    void stopSaverCallback(const Bool::ConstPtr& msg);
    void checkFolder(string save_dir);
    void colorize_disparity(const cv::Mat& src, cv::Mat& dst, int disp_size, cv::InputArray mask);
    ~stereoROS();
};
