
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
    private_nh.getParam("p1", p1);
    private_nh.getParam("p2", p2);
    private_nh.getParam("disp_size", disp_size);
    private_nh.getParam("scale", scale);
    private_nh.getParam("offset_x", offset_x);
    private_nh.getParam("offset_y", offset_y);
    private_nh.getParam("crop_w", crop_w);
    private_nh.getParam("crop_h", crop_h);

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

void stereoROS::imageCb(const ImageConstPtr& l_image_msg, const ImageConstPtr& r_image_msg){
    // std::cout << "start callback" << std::endl;

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

    int height = crop_h * scale;
    int width = crop_w * scale;
    cv::Rect rect(offset_x, offset_y, crop_w, crop_h);  
    cv::resize(l_image(rect), l_image, cv::Size(width, height));
    cv::resize(r_image(rect), r_image, cv::Size(width, height));
    std::cout << "Height and width after crop and resize: " << height << ", " << width << std::endl;

    if (view_input == true){
        cv::imshow("left image", l_image);
        cv::imshow("right image", r_image);
        cv::waitKey(1);
    }

    clock_t start = clock();



    clock_t finish = clock();
    double duration = (double)(finish - start) / CLOCKS_PER_SEC;
    std::cout << "disp estimation used " << duration << " seconds" << endl;

    if (view_output == true){
        cv::imshow("Disparity", disp);
        cv::waitKey(1);
    }

    cv::Mat_<float> disp(height, width);
    cv_bridge::CvImage dispImage;
    dispImage.header = l_image_msg->header;
    dispImage.header.stamp = ros::Time::now();
    dispImage.header.frame_id = "camera_link";
    dispImage.encoding = "32FC1";  // "16UC1";  //
    dispImage.image = disp; // disp * 1000;  //
    pub_disp_.publish(*dispImage.toImageMsg());

    if (save_or_not){
        string left_name = left_dir + time_msg_str + img_suffix;
        cv::imwrite(left_name, l_image);
        std::cout << "Saved " << left_name << std::endl;

        string right_name = right_dir + time_msg_str + img_suffix;
        cv::imwrite(right_name, r_image);
        std::cout << "Saved " << right_name << std::endl;

        string disp_name = disp_dir + time_msg_str + img_suffix;
        cv::Mat disp_16 = cv::Mat(disp.rows, disp.cols, CV_16UC1);
        disp.convertTo(disp_16, CV_16UC1, 1000);
        cv::imwrite(disp_name, disp_16);
        std::cout << "Saved " << disp_name << std::endl;

    }
    clock_t publish_finish = clock();
    double publish_finish_duration = (double)(publish_finish - start) / CLOCKS_PER_SEC;
    std::cout << "disp estimation and publish used " << publish_finish_duration << " seconds" << endl;

    std::cout << "end callback" << std::endl;
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


