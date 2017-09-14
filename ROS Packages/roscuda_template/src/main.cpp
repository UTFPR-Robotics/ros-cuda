#include "ros/ros.h"

void testmain(int size, int *c);

int main(int argc, char **argv)
{
	ros::init(argc, argv, "roscuda_template");	  
	ros::NodeHandle n;
	ros::Rate loop_rate(5);
	int num = 1;
	int *p;
	int s = num*sizeof(int);
	p = (int *)malloc(s);
	testmain(s,p);
	printf("%d\n",p[0]);

	while(ros::ok())
	{    
	    ros::spinOnce();
	    loop_rate.sleep();
	}
	free(p);
	return 0;
}
