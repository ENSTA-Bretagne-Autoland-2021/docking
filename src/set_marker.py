# coding=utf-8
import rospy
import rospkg
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from tf.transformations import euler_from_quaternion, quaternion_from_euler




def set_marker(goal_x, goal_y,goal_z,goal_thetax,goal_thetay,goal_thetaz,objeto):
        state_msg = ModelState()

        state_msg.model_name = objeto
        state_msg.pose.position.x = goal_x
        state_msg.pose.position.y = goal_y
        state_msg.pose.position.z = goal_z
        quaternion = quaternion_from_euler(goal_thetax,goal_thetay,goal_thetaz)
        #type(pose) = geometry_msgs.msg.Pose
        state_msg.pose.orientation.x = quaternion[0]
        state_msg.pose.orientation.y = quaternion[1]
        state_msg.pose.orientation.z = quaternion[2]
        state_msg.pose.orientation.w = quaternion[3]


        
        

        rospy.wait_for_service('/gazebo/set_model_state')
        
        set_state = rospy.ServiceProxy(
            '/gazebo/set_model_state', SetModelState)
        resp = set_state(state_msg)
        #print(state_msg)

