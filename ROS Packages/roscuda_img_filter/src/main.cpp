#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

static const std::string OPENCV_WINDOW_1 = "Image window";
static const std::string OPENCV_WINDOW_2 = "Gray image window";
static const std::string OPENCV_WINDOW_3 = "Filtered image window";
int *difffilter(cv::Mat src, cv::Mat dst);

//IMAGE IS 640 x 480

class ImageConverter
{
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  image_transport::Publisher gimage_pub_;
  image_transport::Publisher image_pub_;

public:
  ImageConverter()
    : it_(nh_)
  {
    // Subscrive to input video feed and publish output video feed
    image_sub_ = it_.subscribe("/image/cam", 1,
      &ImageConverter::imageCb, this);
    gimage_pub_ = it_.advertise("/image_converter/gray_video", 1);
    image_pub_ = it_.advertise("/image_converter/output_video", 1);

    cv::namedWindow(OPENCV_WINDOW_1);
    cv::namedWindow(OPENCV_WINDOW_2);
    cv::namedWindow(OPENCV_WINDOW_3);
  }

  ~ImageConverter()
  {
    cv::destroyWindow(OPENCV_WINDOW_1);
    cv::destroyWindow(OPENCV_WINDOW_2);
    cv::destroyWindow(OPENCV_WINDOW_3);
  }

  void imageCb(const sensor_msgs::ImageConstPtr& msg)
  {
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

    cv::Mat msg_conv, destiny, gray;

    cv::imshow(OPENCV_WINDOW_1, cv_ptr->image);
    cv::waitKey(3);

    cv::cvtColor(cv_ptr->image,msg_conv,CV_BGR2GRAY);
    cv_ptr->image = msg_conv;

    sensor_msgs::ImagePtr gray_msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", cv_ptr->image).toImageMsg();
    destiny = cv::Mat::zeros( cv_ptr->image.size(), CV_8UC1);

    difffilter(cv_ptr->image,destiny);

    // Draw an example circle on the video stream
    //if (cv_ptr->image.rows > 60 && cv_ptr->image.cols > 60)
    //cv::circle(cv_ptr->image, cv::Point(50, 50), 10, CV_RGB(255,0,0));

    // Update GUI Window
    cv::imshow(OPENCV_WINDOW_2, cv_ptr->image);
    cv::waitKey(3);

    // Output modified video stream
    sensor_msgs::ImagePtr dst_msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", destiny).toImageMsg();
    cv::imshow(OPENCV_WINDOW_3, destiny);
    cv::waitKey(3);
    image_pub_.publish(dst_msg);
    gimage_pub_.publish(gray_msg);
  }
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "roscuda_img_filter");
  ImageConverter ic;
  ros::spin();
  return 0;
}