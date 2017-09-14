#include "ros/ros.h"
#include "sensor_msgs/LaserScan.h"
#include "std_msgs/Float32.h"

//Since we already know the laserscan used has fixed size, we will not bother using dynamic size
#define SIZE 511

float *average3(int num, float *a, float *b, float *c, float *d);
float a[SIZE], b[SIZE], c[SIZE], d[SIZE];
sensor_msgs::LaserScan msg_laser;

void laserscan_Callback(const sensor_msgs::LaserScan& msg)
{		
	msg_laser=msg;
	// Cicle vectors (c=b, b=a, a=new)
	for(int i=0;i<SIZE;i++)
	{
		c[i]=b[i];
		b[i]=a[i];
		a[i]=msg.ranges[i];
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
		a[i]=b[i]=c[i]=d[i]=0;
	}
	while(ros::ok())
	{    
		ros::spinOnce();
		average3(SIZE, a, b, c, d);
		// Assign frame_id and ranges size to be able to publish and visualize topic
		msg_laser.header.frame_id="LaserScanner_2D";
		msg_laser.ranges.resize(511);
		// Assign values
		for(int i=0;i<SIZE;i++)
		{
			msg_laser.ranges[i]=d[i];
		}
		laser_pub.publish(msg_laser);
	}
	return 0;
}
