import random
import imageio
import datetime
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Lambda, Concatenate, GRU
from tensorflow.keras.optimizers import SGD,Adam,Adadelta,Adamax,RMSprop
from tensorflow.keras import initializers

from TF2_DDPG_Basic import OrnsteinUhlenbeckNoise, NormalNoise

# Original paper: https://arxiv.org/pdf/1509.02971.pdf

tf.keras.backend.set_floatx('float64')


def actor(input_shape, action_dim, action_bound, action_shift, units=(24, 16)):
	states = Input(shape=input_shape)
	x = LSTM(units[0], name="state_lstm", activation="tanh",kernel_initializer=initializers.RandomNormal(stddev=0.001))(states)
	for index in range(1, len(units)):
		x = Dense(units[index], name="fc{}".format(index), activation='tanh',kernel_initializer=initializers.RandomNormal(stddev=0.01))(x)

	unscaled_output = Dense(action_dim, name="output", activation='tanh',kernel_initializer=initializers.RandomNormal(stddev=0.01))(x)
	scalar = action_bound * np.ones(action_dim)
	output = Lambda(lambda op: op * scalar)(unscaled_output)
	if np.sum(action_shift) != 0:
		output = Lambda(lambda op: op + action_shift)(output)  # for action range not centered at zero

	model = Model(inputs=states, outputs=output)

	return model


def critic(input_shape, action_dim, units=(24, 16)):
	inputs = [Input(shape=input_shape), Input(shape=(action_dim,))]
	state_features = LSTM(units[0], name="state_lstm", activation="tanh",kernel_initializer=initializers.RandomNormal(stddev=0.001))(inputs[0])
	concat = Concatenate(axis=-1)([state_features]+inputs[1:])
	x = Dense(units[1], name="fc1", activation='relu',kernel_initializer=initializers.RandomNormal(stddev=0.01))(concat)
	for index in range(2, len(units)):
		x = Dense(units[index], name="fc{}".format(index), activation='relu',kernel_initializer=initializers.RandomNormal(stddev=0.01))(x)
	output = Dense(1, name="output",kernel_initializer=initializers.RandomNormal(stddev=0.01))(x)
	model = Model(inputs=inputs, outputs=output)

	return model


def update_target_weights(model, target_model, tau=0.005):
	weights = model.get_weights()
	target_weights = target_model.get_weights()
	for i in range(len(target_weights)):  # set tau% of target model to be new weights
		target_weights[i] = weights[i] * tau + target_weights[i] * (1 - tau)
	target_model.set_weights(target_weights)


class DDPG:
	def __init__(
			self,
			observation_space,
			action_space,
			action_space_high,
			action_space_low,
			discrete=False,
			time_steps=3,
			lr_actor=1e-4,
			lr_critic=1e-4,
			actor_units=(64, 64),
			critic_units=(64, 16),
			noise='norm',
			sigma=1,
			tau=0.125,
			gamma=0.99,
			batch_size=1000,
			memory_cap=10000
	):
		self.state_shape = observation_space  # shape of observations
		self.action_dim = action_space   # number of actions
		self.time_steps = time_steps  # number of states (current + past) as input
		#self.action_dim = env.action_space.n if discrete else env.action_space.shape[0]  # number of actions
		input_shape = (self.time_steps, self.state_shape[0])
		self.stored_states = np.zeros(input_shape)
		self.discrete = discrete
		self.action_bound = (action_space_high - action_space_low) / 2 if not discrete else 1.
		self.action_shift = (action_space_high + action_space_low) / 2 if not discrete else 0.
		self.memory = deque(maxlen=memory_cap)
		if noise == 'ou':
			self.noise = OrnsteinUhlenbeckNoise(mu=np.zeros(self.action_dim), sigma=sigma)
		else:
			self.noise = NormalNoise(mu=np.zeros(self.action_dim), sigma=sigma)

		# Define and initialize Actor network
		self.actor = actor(input_shape, self.action_dim, self.action_bound, self.action_shift, actor_units)
		self.actor_target = actor(input_shape, self.action_dim, self.action_bound, self.action_shift, actor_units)
		self.actor_optimizer = RMSprop(learning_rate=lr_actor)
		update_target_weights(self.actor, self.actor_target, tau=1.)

		# Define and initialize Critic network
		self.critic = critic(input_shape, self.action_dim, critic_units)
		self.critic_target = critic(input_shape, self.action_dim, critic_units)
		self.critic_optimizer = RMSprop(learning_rate=lr_critic)
		self.critic.compile(loss="mean_squared_error", optimizer=self.critic_optimizer)
		update_target_weights(self.critic, self.critic_target, tau=1.)

		# Set hyperparameters
		self.gamma = gamma  # discount factor
		self.tau = tau  # target model update
		self.batch_size = batch_size
		self.n_step=5


		# Tensorboard
		self.summaries = {}

	def update_states(self, new_state):
		# move the oldest state to the end of array and replace with new state
		self.stored_states = np.roll(self.stored_states, -1, axis=0)
		self.stored_states[-1] = new_state

	def act(self, add_noise=True):
		states = np.expand_dims(self.stored_states, axis=0).astype(np.float32)
		a = self.actor.predict(states)
		self.summaries["action_x"]=a[0][0]
		self.summaries["action_y"]=a[0][1]
		self.summaries["action_z"]=a[0][2]
		a += self.noise() * add_noise * self.action_bound
		a = tf.clip_by_value(a, -self.action_bound + self.action_shift, self.action_bound + self.action_shift)

		self.summaries['q_val'] = self.critic.predict([states, a])[0][0]

		return a

	def save_model(self, a_fn, c_fn):
		self.actor.save(a_fn)
		self.critic.save(c_fn)

	def load_actor(self, a_fn):
		self.actor.load_weights(a_fn)
		self.actor_target.load_weights(a_fn)
		print(self.actor.summary())

	def load_critic(self, c_fn):
		self.critic.load_weights(c_fn)
		self.critic_target.load_weights(c_fn)
		print(self.critic.summary())

	def remember(self, state, action, reward, next_state, done):
		state = np.expand_dims(state, axis=0)
		next_state = np.expand_dims(next_state, axis=0)
		self.memory.append([state, action, reward, next_state, done])

	def replay(self):
		if len(self.memory) < 100:
			return

		samples=[]
		n=len(self.memory)-(len(self.memory)%self.n_step)
		L=list(self.memory.copy())[-n:]
		#print(len(L))
		#print(n)
		split =np.split(np.array(L),len(self.memory)//self.n_step)
		if len(self.memory)> self.batch_size:
			split=random.sample(split,self.batch_size//self.n_step)
		#print(len(split))
		for s in split:
			cumul_reward=0
			#print("old",s[0][2])
			#print(len(s))
			for step in reversed(s):
				cumul_reward=step[2]+self.gamma*cumul_reward
			s[0][2]=cumul_reward
			#print("new",s[0][2])
			samples.append(s[0])
		#print(len(samples))
		ISWeights = 1.0
		#samples = random.sample(self.memory, self.batch_size)
		s = np.array(samples).T
		states, actions, rewards, next_states, dones = [np.vstack(s[i, :]).astype(np.float) for i in range(5)]
		next_actions = self.actor_target.predict(next_states)
		q_future = self.critic_target.predict([next_states, next_actions])
		target_qs = rewards + q_future * self.gamma * (1. - dones)

		# train critic
		with tf.GradientTape() as tape:
			q_values = self.critic([states, actions])
			td_error = q_values - target_qs
			critic_loss = tf.reduce_mean(tf.math.square(td_error))
		
		
		critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)  
		critic_grad_clip, _ = tf.clip_by_global_norm(critic_grad, 1e10)
		self.summaries['critic_grad']=critic_grad_clip

		self.critic_optimizer.apply_gradients(zip(critic_grad_clip, self.critic.trainable_variables))
	   

		# train actor
		with tf.GradientTape() as tape:
			actions = self.actor(states)
			actor_loss = -tf.reduce_mean(self.critic([states, actions]))

		actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)  # compute actor gradient

		actor_grad_clip, _ = tf.clip_by_global_norm(actor_grad, 1e10)
		self.summaries['actor_grad']=actor_grad_clip


		self.actor_optimizer.apply_gradients(zip(actor_grad_clip, self.actor.trainable_variables))

		# tensorboard info
		self.summaries['critic_loss'] = critic_loss
		self.summaries['actor_loss'] = actor_loss

	def train(self, max_episodes=50, max_epochs=8000, max_steps=500, save_freq=50):
		current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
		train_log_dir = 'logs/DDPG_lstm_' + current_time
		summary_writer = tf.summary.create_file_writer(train_log_dir)

		done, episode, steps, epoch, total_reward = False, 0, 0, 0, 0
		#cur_state = self.env.reset()
		self.update_states(cur_state)  # update stored states
		while episode < max_episodes or epoch < max_epochs:
			if done:
				episode += 1
				print("episode {}: {} total reward, {} steps, {} epochs".format(
					episode, total_reward, steps, epoch))

				with summary_writer.as_default():
					tf.summary.scalar('Main/episode_reward', total_reward, step=episode)
					tf.summary.scalar('Main/episode_steps', steps, step=episode)

				summary_writer.flush()
				self.noise.reset()

				if steps >= max_steps:
					print("episode {}, reached max steps".format(episode))
					self.save_model("ddpg_actor_episode{}.h5".format(episode),
									"ddpg_critic_episode{}.h5".format(episode))

				#done, cur_state, steps, total_reward = False, self.env.reset(), 0, 0
				if episode % save_freq == 0:
					self.save_model("ddpg_actor_episode{}.h5".format(episode),
									"ddpg_critic_episode{}.h5".format(episode))

			a = self.act()  # model determine action, states taken from self.stored_states
			action = np.argmax(a) if self.discrete else a[0]  # post process for discrete action space
			next_state, reward, done, _ = self.env.step(action)  # perform action on env
			cur_stored_states = self.stored_states
			self.update_states(next_state)  # update stored states

			self.remember(cur_stored_states, a, reward, self.stored_states, done)  # add to memory
			self.replay()  # train models through memory replay

			update_target_weights(self.actor, self.actor_target, tau=self.tau)  # iterates target model
			update_target_weights(self.critic, self.critic_target, tau=self.tau)

			total_reward += reward
			steps += 1
			epoch += 1

			# Tensorboard update
			with summary_writer.as_default():
				if len(self.memory) > self.batch_size:
					tf.summary.scalar('Loss/actor_loss', self.summaries['actor_loss'], step=epoch)
					tf.summary.scalar('Loss/critic_loss', self.summaries['critic_loss'], step=epoch)
				tf.summary.scalar('Stats/q_val', self.summaries['q_val'], step=epoch)
				tf.summary.scalar('Main/step_reward', reward, step=epoch)

			summary_writer.flush()

		self.save_model("ddpg_actor_final_episode{}.h5".format(episode),
						"ddpg_critic_final_episode{}.h5".format(episode))

	""" def test(self, render=True, fps=30, filename='test_render.mp4'):
				#cur_state, done, rewards = self.env.reset(), False, 0
				video = imageio.get_writer(filename, fps=fps)
				while not done:
						a, _ = self.act(add_noise=False)
						action = np.argmax(a) if self.discrete else a[0]  # post process for discrete action space
						#next_state, reward, done, _ = self.env.step(action)
						self.update_states(next_state)
						rewards += reward
						if render:
								#video.append_data(self.env.render(mode='rgb_array'))
				video.close()
				return rewards
 """

if __name__ == "__main__":
	gym_env = gym.make("CartPole-v1")
	try:
		# Ensure action bound is symmetric
		assert (gym_env.action_space.high == -gym_env.action_space.low)
		is_discrete = False
		print('Continuous Action Space')
	except AttributeError:
		is_discrete = True
		print('Discrete Action Space')

	ddpg = DDPG(gym_env, discrete=is_discrete, time_steps=5)
	# ddpg.load_critic("lstm_models/time_step5/ddpg_critic_final_episode215.h5")
	# ddpg.load_actor("lstm_models/time_step5/ddpg_actor_final_episode215.h5")
	ddpg.train(max_episodes=50)
	# rewards = ddpg.test()
	# print("Total rewards: ", rewards)

