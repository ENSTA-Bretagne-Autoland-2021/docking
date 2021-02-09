#!/usr/bin/env python
# license removed for brevity
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Vector3Stamped
from nav_msgs.msg import Odometry
import os
from set_position import set_item
from set_marker import set_marker
from visualization_msgs.msg import Marker
import numpy as np
from TF2_DDPG_CONV import *
import cv2
from cv_bridge import CvBridge

global drone_position 
global drone_angular_velocity
global im 
global frame
global bridge

bridge = CvBridge()
drone_position = PoseStamped()
drone_angular_velocity= Odometry()
im = Image()

topic = 'target_marker'



def callback(data):
    
    global drone_position 
    drone_position = data

def callback2(data):
    
    global drone_angular_velocity 
    drone_angular_velocity = data

def callback3(data):
    global im
    global frame
    im = data
    frame=bridge.imgmsg_to_cv2(im, desired_encoding='mono8')
    # resizing image
    #percent by which the image is resized
    scale_percent = 30
    #calculate the 50 percent of original dimensions
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    # dsize
    dsize = (width, height)
    # resize image
    frame = cv2.resize(frame, dsize)
    


def compute_state(wpt, pose,velocity):
    vx=velocity.twist.twist.linear.x
    vy=velocity.twist.twist.linear.y
    vz=velocity.twist.twist.linear.z

    state = np.array([ wpt[0] - pose.pose.position.x,wpt[1] - pose.pose.position.y, wpt[2] - pose.pose.position.z, -wpt[0] + pose.pose.position.x,-wpt[1] + pose.pose.position.y, -wpt[2] + pose.pose.position.z])
    return state.flatten()

def compute_dist(wpt, pose):
    dist =np.sqrt( (wpt[0] - pose.pose.position.x)**2+(wpt[1] - pose.pose.position.y)**2+(wpt[2] - pose.pose.position.z)**2)
    return dist


def compute_distXY(wpt, pose):
    dist =np.sqrt( (wpt[0] - pose.pose.position.x)**2+(wpt[1] - pose.pose.position.y)**2)
    return dist

def compute_distZ(wpt, pose):
    dist =np.sqrt( (wpt[2] - pose.pose.position.z)**2)
    return dist
##################################################################Create HECTOR##################################################################

observation_space=(144,192,1)
action_space=3
action_space_high=5
action_space_low=-5
Hector=DDPG_CONV(observation_space,action_space,action_space_high,action_space_low)
#Hector.load_actor("/home/paul-antoine/workspaceRos/src/docking/tensor_models_conv/hector_actor/variables/variables")
#Hector.load_critic("/home/paul-antoine/workspaceRos/src/docking/tensor_models_conv/hector_critic/variables/variables")
reached=0
if __name__ == '__main__':
    try:
        rospy.init_node('Reinforcement_convolution_node', anonymous=True)
        #subscriber
        rospy.Subscriber("/ground_truth_to_tf/pose", PoseStamped, callback)
        rospy.Subscriber("/ground_truth_to_tf/state", Vector3Stamped, callback2)
        rospy.Subscriber("/downward_cam/camera/image", Image, callback3)

        #Publisher
        pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        marker_pub = rospy.Publisher(topic, Marker)
        cmd_vel = Twist()
        ground_truth= Vector3Stamped()

        #time variable
        rate = rospy.Rate(1000) # 10hz
        t=0
        dt=0.01

        #wait for new state
        rospy.wait_for_message("/downward_cam/camera/image", PoseStamped)
        rospy.wait_for_message("/ground_truth_to_tf/pose", PoseStamped)
        waypoint = (0,0,1.3)
        set_item(waypoint[0],waypoint[1],waypoint[2]+random.randint(4,6),'quadrotor')
            
        #init training variables
        done, episode, steps, epoch, total_reward = False, 0, 0, 0, 0
        cur_state = frame
        last_dist=compute_dist(waypoint, drone_position)


        while not rospy.is_shutdown():
#############################################################Target############################################################# 
            waypoint_marker = (10*np.sin(t),10*np.sin(t),0.5)
            #set with gazebo service
            set_marker(waypoint_marker[0],waypoint_marker[1],waypoint_marker[2],-1.59594, -1.57079, 1.59613,'marker_cube')
            waypoint = (10*np.sin(t),10*np.sin(t),1.3)

################################################################Reinforcement################################################################      
            #take action#
            a = Hector.act(cur_state)  # model determine action given state
            action = a[0]  # post process for discrete action space

            cmd_vel.linear.x =action[0]
            cmd_vel.linear.y =action[1]
            cmd_vel.linear.z = action[2]

            #Sending command
            pub.publish(cmd_vel)

            #wait for new state
            rospy.wait_for_message("/ground_truth_to_tf/pose", PoseStamped)
            rospy.wait_for_message("/downward_cam/camera/image", Image)
            if(frame.shape[0]!=144 or frame.shape[1]!=192):
                while frame.shape[0]!=144 or frame.shape[1]!=192:
                    rospy.wait_for_message("/ground_truth_to_tf/pose", PoseStamped)
                    rospy.wait_for_message("/downward_cam/camera/image", Image)
                    print("error")
            #new_state
            next_state=frame
            #Compute reward
            reward=-50*(compute_dist(waypoint,drone_position)-last_dist)


            if done:
                done=False
                reward=0

            last_dist=compute_dist(waypoint,drone_position)
            #check if it's done
            if compute_distXY(waypoint_marker,drone_position)/drone_position.pose.position.z>0.5 or last_dist>6:
                set_item(waypoint[0],waypoint[1],waypoint[2]+random.randint(3,6),'quadrotor')
                reward=-100
                done=True

            if compute_distXY(wpt, drone_position)<0.2 and compute_distZ(wpt, drone_position)<0.2:
                reached+=1
                done=True
                print("reached")
                print(reached)

            #Training process
            Hector.remember(cur_state, a, reward, next_state, done)  # add to memory
            Hector.replay()  # train models through memory replay

            update_target_weights(Hector.actor, Hector.actor_target, tau=Hector.tau)  # iterates target model
            update_target_weights(Hector.critic, Hector.critic_target, tau=Hector.tau)

            cur_state = next_state
            total_reward += reward
            steps += 1
            epoch += 1
        
            print(reward)
            #update time
            t+=dt
            if reached >=50:
                Hector.save_model("/home/paul-antoine/workspaceRos/src/docking/tensor_models_conv/hector_actor","/home/paul-antoine/workspaceRos/src/docking/tensor_models_conv/hector_critic")
                break
        
    except rospy.ROSInterruptException:
        pass
