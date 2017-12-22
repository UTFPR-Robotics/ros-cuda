#include "ros/ros.h"
#include "sensor_msgs/LaserScan.h"
#include "std_msgs/Float32.h"

//Since we already know the laserscan used has fixed size, we will not bother using dynamic size
#define SIZE 511

float *average3(int num, float *in1, float *in2, float *in3, float *out);
float in1[SIZE], in2[SIZE], in3[SIZE], out[SIZE];
sensor_msgs::LaserScan msg_laser;

void laserscan_Callback(const sensor_msgs::LaserScan& msg)
{		
	msg_laser=msg;
	// Cycle vectors (in3=in2, in2=in1, in1=new)
	for(int i=0;i<SIZE;i++)
	{
		in3[i]=in2[i];
		in2[i]=in1[i];
		in1[i]=msg.ranges[i];
	}
}

int main(int argc, char **argv)
{
	ros::init(argc, argv, "roscuda_laserscan_filter");	  
	ros::NodeHandle n;
	ros::Publisher laser_pub = n.advertise<sensor_msgs::LaserScan>("/laserscan/filtered", 1);
	ros::Subscriber laser_sub = n.subscribe("/laserscan/raw", 100, laserscan_Callback);

	// Initializes the vectors with zeros
	for (int i = 0; i < SIZE; ++i)
	{
		in1[i]=in2[i]=in3[i]=out[i]=0;
	}
	while(ros::ok())
	{   // Get new message and perform average
		ros::spinOnce();
		average3(SIZE, in1, in2, in3, out);
		// Assign frame_id and ranges size to be able to publish and visualize topic
		msg_laser.header.frame_id="LaserScanner_2D";
		msg_laser.ranges.resize(511);
		// Assign values
		for(int i=0;i<SIZE;i++)
		{
			msg_laser.ranges[i]=out[i];	
		}
		laser_pub.publish(msg_laser);
	}
	return 0;
}
