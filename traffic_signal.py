import simpy
import random
import sys

import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop

RANDOM_SEED = 36
qwer = 0

class Intersection(object):
	"""
		Defines the intersections for our traffic simulation
	"""
	def __init__(self, env, id_num):
		self.env = env
		self.id_num = id_num
		self.signal = [env.event() for i in range(8)]
		self.phases = [(1, 5), (1, 6), (2, 5), (2, 6), (3, 7), (3, 8), (4, 7), (4, 8)]
		self.waiting_cars = [[] for i in range(8)]
		self.phase_counter = 0
		self.change_duration = 2
		self.process = env.process(self.run())

		# used only in intersection 5
		self.model = None
		self.rms = None
		self.mode = 0
		self.entire_delay = 0
		self.qval = None
		self.state = None
		if id_num == 5:
			self.model = Sequential()
			self.model.add(Dense(25, init='lecun_uniform', input_shape=(40,)))
			self.model.add(Activation('relu'))
			self.model.add(Dense(8, init='lecun_uniform'))
			self.model.add(Activation('linear'))
			self.rms = RMSprop()
			self.model.compile(loss='mse', optimizer=self.rms)

	def run(self):
		while True:
			current_phase = self.decide_phase(self.id_num)
			lane_num1 = current_phase[0]-1
			lane_num2 = current_phase[1]-1
			# print('---------')
			# print('Time: ',self.env.now)
			# print('Lane num1', lane_num1+1)
			# print('Lane num2', lane_num2+1)
			yield self.env.timeout(self.change_duration)
			self.signal[lane_num1].succeed()
			self.signal[lane_num2].succeed()
			self.signal[lane_num1] = self.env.event()
			self.signal[lane_num2] = self.env.event()
			self.phase_counter = (self.phase_counter + 1)%8
			# print('* wait list:', self.waiting_cars[lane_num1], 'for lane', lane_num1+1,'for intersection id:',self.id_num)
			# print('* wait list:', self.waiting_cars[lane_num2], 'for lane', lane_num2+1,'for intersection id:',self.id_num)
			del self.waiting_cars[lane_num1][:]
			del self.waiting_cars[lane_num2][:]
			self.waiting_cars[lane_num1] = []
			self.waiting_cars[lane_num2] = []
			# print('Deleted waiting_cars lane:', lane_num1+1, 'intersection: ', self.id_num)
			# print('Deleted waiting_cars lane:', lane_num2+1, 'intersection: ', self.id_num)
			# total_delay = self.total_delay()
			# if total_delay != 0:
			# 	print('Total delay:', self.total_delay(),' for intersection', self.id_num)
			# 	print('cars which have left:', qwer)

	def lane_delay(self, lane_id):
		current_time = env.now
		num_cars = len(self.waiting_cars[lane_id - 1])
		# print('Lane:', lane_id, 'at intersection:', self.id_num)
		# print('num cars', num_cars)
		# if (lane_id == 5) and num_cars != 0:
		# 	print('cars:', self.waiting_cars)
		# elif lane_id == 5:
		# 	print('0 cars')
		delay = 0
		for i in range(num_cars):
			# print('self waiting time i:',i,'is',self.waiting_cars[lane_id - 1][i],'fir lane:',lane_id,'at intersection:', self.id_num)
			delay += (current_time - self.waiting_cars[lane_id - 1][i])
		# print('returning delay', delay)
		return delay

	def total_delay(self):
		total_delay = 0
		for i in range(1,9):
			total_delay += self.lane_delay(i)
		return total_delay

	def relative_traffic_flow(self, lane_id):
		lane_delay = self.lane_delay(lane_id)
		total_delay = self.total_delay()
		rel_flow = 0
		if total_delay != 0:
			rel_flow = lane_delay/total_delay
		return rel_flow

	def decide_phase(self, id):
		if id == 5:
			return self.decide_phase_central()
		else:
			return self.decide_phase_outbound()

	def decide_phase_central(self):
		# learn and decide a phase strategy
		epsilon = 0.1
		gamma = 0.95
		if self.env.now <= 3:
			return self.decide_phase_outbound()
		if self.state == None:
			self.state = current_state()
		while True:
			if self.mode == 0:
				print('Entered mode 0')
				self.qval = self.model.predict(np.reshape(self.state, (1, 40)), batch_size=1)
				print('Qval:', self.qval)
				print('***** State:', self.state)
				if (random.random() < epsilon):
					self.action = np.random.randint(0,7)
				else:
					self.action = np.argmax(self.qval)
				print('Action:', self.action)
				self.mode = (self.mode + 1)%2
				return self.phases[self.action]		# take action and observe the world
			elif self.mode == 1:
				print('Entered mode 1')
				new_state = current_state()
				newQ = self.model.predict(np.reshape(new_state, (1, 40)), batch_size=1)
				maxQ = np.max(newQ)
				new_entire_delay = entire_delay()
				print('Delay:', new_entire_delay)
				numerator = self.entire_delay - new_entire_delay
				denominator = max(self.entire_delay, new_entire_delay)
				if denominator == 0:
					print('::Denominator is 0::')
					numerator = 0
					denominator = 1
				reward = numerator/denominator
				self.entire_delay = new_entire_delay
				target = reward + (gamma * maxQ)
				y = np.zeros((1, 8))
				y[:] = self.qval[:]
				y[0][self.action] = target
				self.model.fit(self.state.reshape(1,40), y, batch_size=1, nb_epoch=1, verbose=1)
				self.mode = (self.mode + 1)%2
				self.state = new_state

	def decide_phase_outbound(self):
		total_delay = self.total_delay()
		# print('Total delay is:', total_delay)
		relative_delays = []
		max_delay = 0
		max_phase = None
		for i in range(1, 9):
			rel_delay = self.lane_delay(i)
			if total_delay != 0:
				# print('Rel delay:', rel_delay)
				rel_delay = rel_delay/total_delay
			else:
				rel_delay = 0
			
			relative_delays.append(rel_delay)
		for phase in self.phases:
			rel_delay = relative_delays[phase[0]-1] + relative_delays[phase[1]-1]
			if rel_delay > max_delay:
				max_delay = rel_delay
				max_phase = phase
		if max_phase == None:
			max_phase = self.phases[random.randint(0,7)]
			# print('Choosing phase randomly:', phase)

		else:
			print('Not random, phase choosen:', max_phase, 'for intersection id', self.id_num)
		return max_phase



def current_state():
	state = np.ones((5, 8))
	for i in range(5):
		for j in range(8):
			intersection = intersections[i]
			rel_flow = intersection.relative_traffic_flow(j + 1)
			state[i][j] = rel_flow
	return state

def entire_delay():
	total_delay = 0
	for i in range(5):
		intersection = intersections[i]
		delay = intersection.total_delay()
		if i == 4:
			delay *= 0.75
		else:
			delay *= 0.25
		total_delay += delay
	return total_delay
		


def car_generator(env, intersection_id, lane_id, intersections, lanes):
	'''
	Generates cars for our simulations at the starting of lane-id at intersection-id
	'''
	counter = 0
	for i in range(10000):
		name = 'I' + str(intersection_id) + 'L' + str(lane_id) + '-' + str(counter)
		c = car(env, name, intersection_id, lane_id, intersections, lanes)
		env.process(c)
		# print('i:',i)
		# print('intersection id:', intersection_id)
		# print('lane_id:', lane_id)
		# print('* New car generated: ', name , 'at time ', env.now)
		counter += 1
		t = random.expovariate(1.0/0.15)
		# print('Timeout', t)
		yield env.timeout(t)
		# print('Time', env.now)


def car(env, name, intersection_id, lane_id, intersections, lanes):
	# Simulating the process of car traversing the lane 
	# print('* intersection id:', intersection_id)
	# print('* lane id:', lane_id)
	global qwer
	arrive = env.now
	# print('* Car: ', name, ' arrives at ', arrive)
	intersection = intersections[intersection_id-1]
	with lanes[intersection_id - 1][lane_id - 1].request() as req:
		yield req
		# print('* Car: ', name, ' traversing lane', lane_id , ' at intersection ID', intersection_id)
		duration = 2
		yield env.timeout(duration)
	current_time = env.now
	# print('waiting car at lane:', lane_id, ' at intersection id:', intersection_id)
	intersection.waiting_cars[lane_id-1].append(current_time)
	# print('wai8:', intersection.waiting_cars[lane_id-1],'for lane:', lane_id, 'for intersection:', intersection_id)
	# print('Car: ', name, 'waiting at signal', 'Time:', env.now)
	# Simulating waiting at the red/green light at the intersection
	temp = env.now
	yield intersection.signal[lane_id - 1]
	# print('Car: ', name, 'signal opened', 'waiting time:', env.now - temp)
	# print('Car: ', name, 'crossing the signal')
	#Simulating time taken to cross intersection
	yield env.timeout(3)

	# Simulating entering the lane
	# print('Car: ', name, ' is now switching lanes ')
	rk = random.randint(1, 6) #if rk = 0, go straight else turn right
	# print('rk 0 :',rk)
	if rk%2 == 0:
		rk = 0
	else:
		rk = 1
	# print('rk:', rk)
	new_intersection_id = 0
	new_lane_id = 0

	if intersection_id == 1:
		if lane_id == 2:
			if rk == 0:
				new_intersection_id = 5
				new_lane_id = 2
			else:
				new_intersection_id = -1
				new_lane_id = 4
				# print('New lane is now 4, leaving curcuit')
		elif lane_id == 4:
			if rk == 0:
				new_intersection_id = -1
				new_lane_id = 4
			else:
				new_intersection_id = -1
				new_lane_id = 6
		#new
		elif lane_id == 6:
			if rk == 0:
				new_intersection_id = -1
				new_lane_id = 6
			else:
				new_intersection_id = -1
				new_lane_id = 8
				
		elif lane_id == 8:
			if rk == 0:
				new_intersection_id = -1
				new_lane_id = 8
			else:
				new_intersection_id = 5
				new_lane_id = 2

		#new
		elif lane_id == 1:
			new_intersection_id = -1
			new_lane_id = 7
		elif lane_id == 3:
			new_intersection_id = -1
			new_lane_id = 1
		elif lane_id == 5:
			new_intersection_id = -1
			new_lane_id = 3
		elif lane_id == 7:
			new_intersection_id = 5
			new_lane_id = 5

		# print('New intersection id:', new_intersection_id)
		# print('New lane id:', new_lane_id)

	elif intersection_id == 2:
		if lane_id == 2:
			if rk == 0:
				new_intersection_id = -1
				new_lane_id = 2
			else:
				new_intersection_id = 5
				new_lane_id = 4
		elif lane_id == 4:
			if rk == 0:
				new_intersection_id = 5
				new_lane_id = 4
			else:
				new_intersection_id = -1
				new_lane_id = 6
		elif lane_id == 6:
			if rk == 0:
				new_intersection_id = -1
				new_lane_id = 6
			else:
				new_intersection_id = -1
				new_lane_id = 8
		#new
		elif lane_id == 8:
			if rk == 0:
				new_intersection_id = -1
				new_lane_id = 8
			else:
				new_intersection_id = -1
				new_lane_id = 2
		elif lane_id == 1:
			new_intersection_id = 5
			new_lane_id = 7
			#new
		elif lane_id == 3:
			new_intersection_id = -1
			new_lane_id = 1
		elif lane_id == 5:
			new_intersection_id = -1
			new_lane_id = 3
		elif lane_id == 7:
			new_intersection_id = -1
			new_lane_id = 5

	elif intersection_id == 3:
		#new
		if lane_id == 2:
			if rk == 0:
				new_intersection_id = -1
				new_lane_id = 2
			else:
				new_intersection_id = -1
				new_lane_id = 4
		elif lane_id == 4:
			if rk == 0:
				new_intersection_id = -1
				new_lane_id = 4
			else:
				new_intersection_id = 5
				new_lane_id = 6
		elif lane_id == 6:
			if rk == 0:
				new_intersection_id = 5
				new_lane_id = 6
			else:
				new_intersection_id = -1
				new_lane_id = 8
		elif lane_id == 8:
			if rk == 0:
				new_intersection_id = -1
				new_lane_id = 8
			else:
				new_intersection_id = -1
				new_lane_id = 2
		elif lane_id == 1:
			new_intersection_id = -1
			new_lane_id = 7
		elif lane_id == 3:
			new_intersection_id = 5
			new_lane_id = 1
		#new
		elif lane_id == 5:
			new_intersection_id = -1
			new_lane_id = 3
		elif lane_id == 7:
			new_intersection_id = -1
			new_lane_id = 5

	elif intersection_id == 4:
		if lane_id == 2:
			if rk == 0:
				new_intersection_id = -1
				new_lane_id = 2
			else:
				new_intersection_id = -1
				new_lane_id = 4
		#new
		elif lane_id == 4:
			if rk == 0:
				new_intersection_id = -1
				new_lane_id = 4
			else:
				new_intersection_id = -1
				new_lane_id = 6

		elif lane_id == 6:
			if rk == 0:
				new_intersection_id = -1
				new_lane_id = 6
			else:
				new_intersection_id = 5
				new_lane_id = 8
		
		elif lane_id == 8:
			if rk == 0:
				new_intersection_id = 5
				new_lane_id = 8
			else:
				new_intersection_id = -1
				new_lane_id = 2
		elif lane_id == 1:
			new_intersection_id = -1
			new_lane_id = 7
		elif lane_id == 3:
			new_intersection_id = -1
			new_lane_id = 1
		elif lane_id == 5:
			new_intersection_id = 5
			new_lane_id = 3
		#new
		elif lane_id == 7:
			new_intersection_id = -1
			new_lane_id = 5

	elif intersection_id == 5:
		if lane_id == 2:
			if rk == 0:
				new_intersection_id = 3
				new_lane_id = 2
			else:
				new_intersection_id = 4
				new_lane_id = 4
		elif lane_id == 4:
			if rk == 0:
				new_intersection_id = 4
				new_lane_id = 4
			else:
				new_intersection_id = 1
				new_lane_id = 6
		elif lane_id == 6:
			if rk == 0:
				new_intersection_id = 1
				new_lane_id = 6
			else:
				new_intersection_id = 2
				new_lane_id = 8
		elif lane_id == 8:
			if rk == 0:
				new_intersection_id = 2
				new_lane_id = 8
			else:
				new_intersection_id = 3
				new_lane_id = 2
		elif lane_id == 1:
			new_intersection_id = 4
			new_lane_id = 7
		elif lane_id == 3:
			new_intersection_id = 1
			new_lane_id = 1
		elif lane_id == 5:
			new_intersection_id = 2
			new_lane_id = 3
		elif lane_id == 7:
			new_intersection_id = 3
			new_lane_id = 5

	# print('new intersection:', new_intersection_id)
	# print('new lane id:', new_lane_id)

	if new_intersection_id != -1:
		env.process(car(env, name, new_intersection_id, new_lane_id, intersections, lanes))
		# print('* Car', name, 'now at intersection:', new_intersection_id,'at lane:',new_lane_id)
	else:
		qwer += 1
		# print('Cars which have left:', qwer)
		# print('Car', name, ' has left')



print('RL based multi-agent system for Network traffic control')
random.seed(RANDOM_SEED)
env = simpy.Environment()

# Adding intersections in the environment
intersections = [Intersection(env, i+1) for i in range(5)]

lanes = [[simpy.Resource(env, capacity=40) for j in range(8)] for i in range(5)]

# Adding car generators 

env.process((car_generator(env, 1, 5, intersections, lanes)))
env.process((car_generator(env, 1, 2, intersections, lanes)))
env.process((car_generator(env, 1, 4, intersections, lanes)))
env.process((car_generator(env, 1, 7, intersections, lanes)))
env.process((car_generator(env, 1, 3, intersections, lanes)))
env.process((car_generator(env, 1, 8, intersections, lanes)))

env.process((car_generator(env, 2, 5, intersections, lanes)))
env.process((car_generator(env, 2, 2, intersections, lanes)))
env.process((car_generator(env, 2, 4, intersections, lanes)))
env.process((car_generator(env, 2, 7, intersections, lanes)))
env.process((car_generator(env, 2, 6, intersections, lanes)))
env.process((car_generator(env, 2, 1, intersections, lanes)))

env.process((car_generator(env, 3, 3, intersections, lanes)))
env.process((car_generator(env, 3, 8, intersections, lanes)))
env.process((car_generator(env, 3, 4, intersections, lanes)))
env.process((car_generator(env, 3, 7, intersections, lanes)))
env.process((car_generator(env, 3, 6, intersections, lanes)))
env.process((car_generator(env, 3, 1, intersections, lanes)))

env.process((car_generator(env, 4, 3, intersections, lanes)))
env.process((car_generator(env, 4, 8, intersections, lanes)))
env.process((car_generator(env, 4, 5, intersections, lanes)))
env.process((car_generator(env, 4, 2, intersections, lanes)))
env.process((car_generator(env, 4, 6, intersections, lanes)))
env.process((car_generator(env, 4, 1, intersections, lanes)))

env.run()
