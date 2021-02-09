#!/usr/bin/env python
# license removed for brevity
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Vector3Stamped
from nav_msgs.msg import Odometry

from set_position import set_item
from visualization_msgs.msg import Marker
import numpy as np
from A2C_continuous import *
import random
drone_position = PoseStamped()
drone_angular_velocity= Odometry()
topic = 'target_marker'



def callback(data):
    
    global drone_position 
    drone_position = data

def callback2(data):
    
    global drone_angular_velocity 
    drone_angular_velocity = data


def compute_state(wpt, pose,velocity):
    vx=velocity.twist.twist.linear.x
    vy=velocity.twist.twist.linear.y
    vz=velocity.twist.twist.linear.z

    state = np.array([wpt[2] - pose.pose.position.z,-wpt[2] + pose.pose.position.z])
    return state.flatten()

def compute_dist(wpt, pose):
    dist =np.sqrt( (wpt[2] - pose.pose.position.z)**2)
    return dist

##################################################################Create HECTOR##################################################################

state_size=2
action_size=1
max_action=10
Hector=ContinuousA2CAgent(action_size, max_action)
scores, episodes = [], []
score_avg = 0



if __name__ == '__main__':
    try:
        rospy.Subscriber("/ground_truth_to_tf/pose", PoseStamped, callback)
        rospy.Subscriber("/ground_truth_to_tf/state", Vector3Stamped, callback2)

        pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        #pub2 = rospy.Publisher('/ground_truth_to_tf/pose', Vector3Stamped, queue_size=10)

        marker_pub = rospy.Publisher(topic, Marker)
        rospy.init_node('Reinforcement_node', anonymous=True)
        rate = rospy.Rate(1000) # 10hz
        cmd_vel = Twist()
        ground_truth= Vector3Stamped()
        dt = 0.1
        

        t = 0.
        rospy.wait_for_message("/ground_truth_to_tf/pose", PoseStamped)
        waypoint = (5,5,random.randint(10,30) )
        set_item(5,5,waypoint[2]+random.randint(-10,10),'quadrotor')
        #launch hector
        cmd_vel.linear.z = 10
        pub.publish(cmd_vel)

        #init training variables
        done = False
        score = 0
        loss_list, sigma_list = [], []
        state = compute_state(waypoint,drone_position,drone_angular_velocity)
        state = np.reshape(state, [1, state_size])
        last_dist=compute_dist(waypoint, drone_position)



        while not rospy.is_shutdown():
#############################################################Way Point############################################################# 
            if last_dist<4:
                waypoint =(5,5,random.randint(10,30) )
                print("reached")
                
            marker = Marker()
            marker.header.frame_id = "world"
            marker.type = marker.SPHERE
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.pose.orientation.w = 1.0
            marker.pose.position.x = waypoint[0]
            marker.pose.position.y = waypoint[1]
            marker.pose.position.z = waypoint[2]
                
################################################################Reinforcement################################################################      
            #take action#
            action = Hector.get_action(state)

            """ cmd_vel.linear.y =action[0]
            cmd_vel.linear.y =action[1] """
            cmd_vel.linear.z = action[0]
            #Sending command
            pub.publish(cmd_vel)
            marker_pub.publish(marker)

            #wait for new state
            rospy.wait_for_message("/ground_truth_to_tf/pose", PoseStamped)
            #new_state
            next_state=compute_state(waypoint,drone_position,drone_angular_velocity)
            next_state = np.reshape(next_state, [1, state_size])

            #Compute reward
            abs_velocity_angular=np.sqrt(drone_angular_velocity.twist.twist.angular.x**2+drone_angular_velocity.twist.twist.angular.y**2+drone_angular_velocity.twist.twist.angular.z**2)
            #reward=-100*(compute_dist(waypoint,drone_position)-last_dist)
            
            if compute_dist(waypoint,drone_position)-last_dist>=0:
                reward=-2
            else:
                reward=+1


            if done:
                done=False
                reward=0
                score = 0
                loss_list, sigma_list = [], []
                

            if drone_position.pose.position.z <0.3:
                reward=-10
            """ else:
                reward=1 """
            last_dist=compute_dist(waypoint,drone_position)

            #check if it's done
            if last_dist>30 or drone_position.pose.position.z <0.3 :
                set_item(5,5,waypoint[2]+random.randint(-10,10),'quadrotor')
                done=True

            #Training process
            score += reward
            #reward = 0.1 if not done or score == 500 else -1

            loss, sigma = Hector.train_model(state, action, reward, next_state, done)
            loss_list.append(loss)
            sigma_list.append(sigma)
            state = next_state

            if done:
                score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
                
                scores.append(score_avg)
                """ episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.xlabel("episode")
                pylab.ylabel("average score")
                pylab.savefig("./save_graph/graph.png") """

                """ if score_avg > 400:
                    agent.model.save_weights("./save_model/model", save_format="tf")
                    sys.exit() """
            print(reward)
    except rospy.ROSInterruptException:
        pass
