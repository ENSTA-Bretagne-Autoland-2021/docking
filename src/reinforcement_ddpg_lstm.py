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
from TF2_DDPG_LSTM import *

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
    
    state = np.array([ wpt[0] - pose.pose.position.x,wpt[1] - pose.pose.position.y, wpt[2] - pose.pose.position.z, -wpt[0] + pose.pose.position.x,-wpt[1] + pose.pose.position.y, -wpt[2] + pose.pose.position.z])

    return state.flatten()

def compute_dist(wpt, pose):
    dist =np.sqrt( (wpt[0] - pose.pose.position.x)**2+(wpt[1] - pose.pose.position.y)**2+(wpt[2] - pose.pose.position.z)**2)
    return dist

##################################################################Create HECTOR##################################################################

observation_space=[6]
action_space=3
action_space_high=5.
action_space_low=-5.
Hector=DDPG(observation_space,action_space,action_space_high,action_space_low)
Hector.load_actor("/home/paul-antoine/workspaceRos/src/docking/tensor_models_lstm/hector_actor/variables/variables")
Hector.load_critic("/home/paul-antoine/workspaceRos/src/docking/tensor_models_lstm/hector_critic/variables/variables")
reached=0
episode=0
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
        waypoint = (random.randint(-10,10), random.randint(-10,10),random.randint(10,30) )
        set_item(waypoint[0]+random.randint(-10,10),waypoint[1]+random.randint(-10,10),waypoint[2]+random.randint(-10,10),'quadrotor')
        

        #init training variables
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/DDPG_basic_lstm' + current_time
        summary_writer = tf.summary.create_file_writer(train_log_dir)

        done, episode, steps, epoch, total_reward = False, 0, 0, 0, 0
        cur_state = compute_state(waypoint,drone_position,drone_angular_velocity)
        Hector.update_states(cur_state)  # update stored states
        last_dist=compute_dist(waypoint, drone_position)



        while not rospy.is_shutdown():
#############################################################Way Point############################################################# 
            if last_dist<3:
                waypoint =(random.randint(-10,10),random.randint(-10,10),random.randint(10,30) )
                reached+=1
                done=True
                print("reached")
                print(reached)

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
            a = Hector.act()  # model determine action given state
            action = a[0]  # post process for discrete action space

            cmd_vel.linear.x =action[0]
            cmd_vel.linear.y =action[1]
            cmd_vel.linear.z = action[2]
            #Sending command
            pub.publish(cmd_vel)
            marker_pub.publish(marker)

            #wait for new state
            rospy.wait_for_message("/ground_truth_to_tf/pose", PoseStamped)
            #new_state
            next_state=compute_state(waypoint,drone_position,drone_angular_velocity)

            #Compute reward
            reward=-50*(compute_dist(waypoint,drone_position)-last_dist)




            if done:
                done=False
                reward=0
            if drone_position.pose.position.z <0.3:
                reward=-100
            """ else:
                reward=1 """
            last_dist=compute_dist(waypoint,drone_position)

            #check if it's done
            if last_dist>20 or drone_position.pose.position.z <0.3 :
                set_item(waypoint[0]+random.randint(-10,10),waypoint[1]+random.randint(-10,10),waypoint[2]+random.randint(-10,10),'quadrotor')
                reward=0
                done=True

            #Training process
           
            cur_stored_states = Hector.stored_states
            Hector.update_states(next_state)  # update stored states

            Hector.remember(cur_stored_states, a, reward, Hector.stored_states, done)  # add to memory
            Hector.replay()  # train models through memory replay

            update_target_weights(Hector.actor, Hector.actor_target, tau=Hector.tau)  # iterates target model
            update_target_weights(Hector.critic, Hector.critic_target, tau=Hector.tau)

            cur_state = next_state
            total_reward += reward
            steps += 1
            epoch += 1

            


            print(reward)
            #rate.sleep()
            t=t+dt
             #end the simulation
            if reached >=50 or episode>10000:
                Hector.save_model("/home/paul-antoine/workspaceRos/src/docking/tensor_models_lstm/hector_actor","/home/paul-antoine/workspaceRos/src/docking/tensor_models_lstm/hector_critic")
                break



            #tensorboard
            episode+=1
        
            with summary_writer.as_default():
                if len(Hector.memory) > 100:
                    tf.summary.scalar('Loss/actor_loss', Hector.summaries['actor_loss'], step=episode)
                    tf.summary.scalar('Loss/critic_loss', Hector.summaries['critic_loss'], step=episode)
                    count=0
                    for t in Hector.summaries['actor_grad']:
                        tf.summary.histogram('actor/'+str(count), data=t,step=episode)
                        count+=1

                    count=0
                    for t in Hector.summaries['critic_grad']:
                        tf.summary.histogram('critic/'+str(count), data=t,step=episode)
                        count+=1
                    """ for index, grad in enumerate(Hector.summaries['critic_grad']): 
                        tf.summary.histogram("{}-grad".format(Hector.summaries['critic_grad'][index][1].name), Hector.summaries['critic_grad'][index]) """ 

                tf.summary.scalar('Action/action_x',Hector.summaries['action_x'],step=episode)
                tf.summary.scalar('Action/action_y',Hector.summaries['action_y'],step=episode)
                tf.summary.scalar('Action/action_z',Hector.summaries['action_z'],step=episode)
                tf.summary.scalar('Main/step_reward', reward, step=episode)
                tf.summary.scalar('Stats/q_val', Hector.summaries['q_val'], step=episode)



    except rospy.ROSInterruptException:
        Hector.save_model("/home/paul-antoine/workspaceRos/src/docking/tensor_models_lstm/hector_actor","/home/paul-antoine/workspaceRos/src/docking/tensor_models_lstm/hector_critic")
        pass
