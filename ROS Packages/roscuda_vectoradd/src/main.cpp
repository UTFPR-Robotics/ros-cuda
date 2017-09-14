#include "ros/ros.h"
#include <time.h>
#include "std_msgs/Int32.h"
#include "std_msgs/Float32.h"


#define N (2048*2048) //max=127*2048*2048
#define THREADS_PER_BLOCK 512

int *testmain(int num, int threads);
int size=10;
std_msgs::Float32 msg_time;

void size_Callback(const std_msgs::Int32& msg)
{
	size=msg.data;
	if(size<1){size=1;}
	if(size>127){size=127;}
}

void printall(int *vec, int tot)
{
	// Print
	printf("(");
	for(int k = 0; k<(tot-1); k++)
	{	
		printf("%d, ",*(vec+k));
	}
	printf("%d)",*(vec+tot));
	fflush(stdout);
}

int main(int argc, char **argv)
{
	ros::init(argc, argv, "beginner_tutorials");	  
	ros::NodeHandle n;
	ros::Publisher time_pub = n.advertise<std_msgs::Float32>("/time", 1);
	ros::Subscriber size_sub = n.subscribe("/size", 100, size_Callback);

	clock_t start, end;
 	double cpu_time_used;
 	
	int *p;

	while(ros::ok())
	{    
		start = clock();

		p = testmain(size*N,THREADS_PER_BLOCK);
		end = clock();
		cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
		msg_time.data=cpu_time_used;
		time_pub.publish(msg_time);
		
		//printall(p,N);
		printf("\nTime taken: %f\n---\n",cpu_time_used);
		fflush(stdout);

	    ros::spinOnce();
	    
	}
	return 0;
}
