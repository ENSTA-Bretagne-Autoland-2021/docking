# coding=utf-8
import rospy
import rospkg
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState




def set_item(goal_x, goal_y,goal_z,objeto):
        state_msg = ModelState()

        state_msg.model_name = objeto
        state_msg.pose.position.x = goal_x
        state_msg.pose.position.y = goal_y
        state_msg.pose.position.z = goal_z
        

        rospy.wait_for_service('/gazebo/set_model_state')
        
        set_state = rospy.ServiceProxy(
            '/gazebo/set_model_state', SetModelState)
        resp = set_state(state_msg)
        #print(state_msg)

