
#include "stereoROS.h"


stereoROS::stereoROS(ros::NodeHandle* nodehandle):nh_(*nodehandle){
    it_.reset(new image_transport::ImageTransport(nh_));

    image_transport::TransportHints hints("raw", ros::TransportHints(), nh_);
    sub_l_image_.subscribe(*it_, "/left/image_rect", 1, hints);
    sub_r_image_.subscribe(*it_, "/right/image_rect", 1, hints);

    // Synchronize inputs. Topic subscriptions happen on demand in the connection
    // callback. Optionally do approximate synchronization.
    ros::NodeHandle private_nh("~");
    private_nh.getParam("queue_size", queue_size);
    private_nh.getParam("approximate_sync", approx);
    private_nh.getParam("save_dir", save_dir);
    private_nh.getParam("save_or_not", save_or_not);
    private_nh.getParam("view_input", view_input);
    private_nh.getParam("view_output", view_output);
    private_nh.getParam("scale", scale);
    // private_nh.getParam("offset_x", offset_x);
    // private_nh.getParam("offset_y", offset_y);
    // private_nh.getParam("width", width);
    // private_nh.getParam("height", height);
    private_nh.getParam("p1", p1);
    private_nh.getParam("p2", p2);
    private_nh.getParam("disp_size", disp_size);
    private_nh.getParam("uniqueness", uniqueness);
    private_nh.getParam("num_paths", num_paths);
    private_nh.getParam("min_disp", min_disp);
    private_nh.getParam("LR_max_diff", LR_max_diff);
    private_nh.getParam("cen", cen);
    private_nh.getParam("bl", bl);
    private_nh.getParam("focal", focal);
    
    stereoROS::checkFolder(save_dir);

    if (approx)
    { 
        approximate_sync_.reset( new ApproximateSync(ApproximatePolicy(queue_size),
                                                        sub_l_image_, sub_r_image_) );
        approximate_sync_->registerCallback(boost::bind(&stereoROS::imageCb,
                                                        this, _1, _2));
    }
    else
    {
        exact_sync_.reset( new ExactSync(ExactPolicy(queue_size),
                                            sub_l_image_, sub_r_image_) );
        exact_sync_->registerCallback(boost::bind(&stereoROS::imageCb,
                                                    this, _1, _2));
    }

    saver_sub_ = nh_.subscribe("save_trigger", 100, &stereoROS::saverCallback, this);
    stop_saver_sub_ = nh_.subscribe("stop_save_trigger", 100, &stereoROS::stopSaverCallback, this);

    // pub_disp_ = nh_.advertise<DisparityImage>("/stereo_ros/disparity", 1);
    pub_disp_ = nh_.advertise<sensor_msgs::Image>("/stereo_ros/disparity", 1);
    pub_depth_ = nh_.advertise<sensor_msgs::Image>("/stereo_ros/depth", 1);

    stereoROS::setParam();
}

void stereoROS::makeOutputFolder(string folderName)
{
    if (!boost::filesystem::exists(folderName))
    {
        if (!boost::filesystem::create_directories(folderName))
        {
            stringstream errorMsg;
            std::cout << "Could not create output directory: " << folderName << std::endl;
        }
    }
}

void stereoROS::setParam(){
	const sgm::PathType path_type = num_paths == 8 ? sgm::PathType::SCAN_8PATH : sgm::PathType::SCAN_4PATH;
    sgm::CensusType census_type = sgm::CensusType(cen);
	const int dst_depth = disp_size < 256 ? 8 : 16;
	const int src_bytes = src_depth * width * height / 8;
	const int dst_bytes = dst_depth * width * height / 8;

	// new_sgm(width, height, disp_size, src_depth, dst_depth, sgm::EXECUTE_INOUT_CUDA2CUDA);

    d_I1.allocate(src_bytes);
    d_I2.allocate(src_bytes);
    d_disparity.allocate(dst_bytes);

	disparity = cv::Mat(height, width, dst_depth == 8 ? CV_8S : CV_16S);
	// invalid_disp = new_sgm.get_invalid_disparity();
    depth = cv::Mat(height, width, CV_32F);

}

void stereoROS::imageCb(const ImageConstPtr& l_image_msg, const ImageConstPtr& r_image_msg){
    // std::cout << "start callback" << std::endl;

    clock_t start = clock();

    cv_bridge::CvImagePtr l_cv_ptr;
    cv_bridge::CvImagePtr r_cv_ptr;
    l_cv_ptr = cv_bridge::toCvCopy(l_image_msg, "mono8");
    r_cv_ptr = cv_bridge::toCvCopy(r_image_msg, "mono8");
    l_cv_ptr->image.copyTo(l_image);
    r_cv_ptr->image.copyTo(r_image);
    ros::Time timeStampMsg;
    timeStampMsg.sec = l_image_msg->header.stamp.sec;
    timeStampMsg.nsec = l_image_msg->header.stamp.nsec;
    string time_msg_str = to_string(timeStampMsg.sec) + "_" + to_string(timeStampMsg.nsec);

    if (l_image.rows == height && l_image.cols == width){
        I1 = l_image;
        I2 = r_image;
    }
    else{
        cv::resize(l_image, I1, cv::Size(width, height));
        cv::resize(r_image, I2, cv::Size(width, height));
        scale = width / l_image.cols;
    }

    d_I1.upload(I1.data);
    d_I2.upload(I2.data);

    sgm::StereoSGM new_sgm(width, height, disp_size, src_depth, dst_depth, sgm::EXECUTE_INOUT_CUDA2CUDA);
    new_sgm.execute(d_I1.data, d_I2.data, d_disparity.data);
    cudaDeviceSynchronize();
	d_disparity.download(disparity.data);
    invalid_disp = new_sgm.get_invalid_disparity();

    #pragma omp parallel for 
    for(int i = 0; i < disparity.rows; ++i)
    {
        for(int j = 0; j < disparity.cols; ++j)
        {
            float d = disparity.at<float>(i, j);
            if(d > 0) // To avoid division by zero
            {
                depth.at<float>(i, j) = (focal * bl) / d;
            }
            else
            {
                depth.at<float>(i, j) = 0;
            }
        }
    }

	// show image
    clock_t finish = clock();
    double duration = (double)(finish - start) / CLOCKS_PER_SEC;
    // std::cout << "disp estimation used " << duration << " seconds" << endl;

    if (view_input == true){
        cv::imshow("left image", I1);
        cv::imshow("right image", I2);
        cv::waitKey(1);
    }

    if (view_output == true){
        cv::imshow("Disparity", disparity);
        cv::waitKey(1);
    }
    cv_bridge::CvImage dispImage;
    dispImage.header = l_image_msg->header;
    dispImage.header.stamp = ros::Time::now();
    dispImage.header.frame_id = l_image_msg->header.frame_id;
    dispImage.encoding = "8UC1"; 
    dispImage.image = disparity; 
    pub_disp_.publish(*dispImage.toImageMsg());

    cv_bridge::CvImage depthImage;
    depthImage.header = l_image_msg->header;
    depthImage.header.stamp = ros::Time::now();
    depthImage.header.frame_id = l_image_msg->header.frame_id;
    depthImage.encoding = "32FC1"; 
    depthImage.image = depth; 
    pub_depth_.publish(*depthImage.toImageMsg());   

    if (save_or_not){
        string left_name = left_dir + time_msg_str + img_suffix;
        cv::imwrite(left_name, l_image);
        std::cout << "Saved " << left_name << std::endl;

        string right_name = right_dir + time_msg_str + img_suffix;
        cv::imwrite(right_name, r_image);
        std::cout << "Saved " << right_name << std::endl;

        string disp_name = disp_dir + time_msg_str + img_suffix;
        cv::Mat disp_16 = cv::Mat(disparity.rows, disparity.cols, CV_16UC1);
        disparity.convertTo(disp_16, CV_16UC1, 1000);
        cv::imwrite(disp_name, disp_16);
        std::cout << "Saved " << disp_name << std::endl;

    }
    clock_t publish_finish = clock();
    double publish_finish_duration = (double)(publish_finish - start) / CLOCKS_PER_SEC;
    // std::cout << "disp estimation and publish used " << publish_finish_duration << " seconds" << endl;

    // std::cout << "end callback" << std::endl;
}

void stereoROS::checkFolder(string save_dir){
    left_dir = save_dir + "left/";
    right_dir = save_dir + "right/";
    disp_dir = save_dir + "disp/";

    if (boost::filesystem::exists(save_dir) == false)
    {   
        std::cout << "not existing " << save_dir << ", creating now" << std::endl;
        boost::filesystem::create_directories(save_dir);
    }
    if (boost::filesystem::exists(left_dir) == false)
    {
        std::cout << "not existing " << left_dir << ", creating now" << std::endl;
        boost::filesystem::create_directories(left_dir);
    }
    if (boost::filesystem::exists(right_dir) == false)
    {
        std::cout << "not existing " << right_dir << ", creating now" << std::endl;
        boost::filesystem::create_directories(right_dir);
    }
    if (boost::filesystem::exists(disp_dir) == false)
    {
        std::cout << "not existing " << disp_dir << ", creating now" << std::endl;
        boost::filesystem::create_directories(disp_dir);
    }
    img_suffix = ".png";
    std::cout << "Save floder check finished! Will save to " << save_dir << std::endl;
}

void stereoROS::saverCallback(const Bool::ConstPtr& msg)
{
  save_or_not = true;
}

void stereoROS::stopSaverCallback(const Bool::ConstPtr& msg)
{
  save_or_not = false;
}
void stereoROS::colorize_disparity(const cv::Mat& src, cv::Mat& dst, int disp_size, cv::InputArray mask)
{
    cv::Mat tmp;
    src.convertTo(tmp, CV_8U, 255. / disp_size);
    cv::applyColorMap(tmp, dst, cv::COLORMAP_TURBO);
    if (!mask.empty())
    dst.setTo(0, mask);
}
stereoROS::~stereoROS(){
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "stereoROS");
  ros::NodeHandle nh;
  stereoROS fd(&nh);
//   stereoROS fd();
  std::cout  << "stereoROS Running" << std::endl;

  ros::spin();
  return EXIT_SUCCESS;
}


