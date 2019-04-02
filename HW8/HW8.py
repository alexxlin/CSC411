import numpy as np

#define the transition probability
P = np.array([[[0, 1], [0, 1]], [[1, 0], [1, 0]]])

#define the immediate reward distribution
R = np.array([[-1, 1], [0, 5]])

def find_optimal(P, R, time_steps, gamma):
	
	states_num = np.shape(P)[0]
	actions_num = np.shape(P)[1]

	Q = np.zeros((states_num, actions_num))

	for i in range(time_steps):

		Q_temp = Q

		for s in range(states_num):
			for a in range(actions_num):

				# Calculate reward for all future states, at current state s and current action a
				optimal_policy = 0
				for s_next in range(states_num):
					index = np.argmax(Q[s_next])
					optimal_policy += P[a,s,s_next] * Q[s_next, index]

				Q_temp[s, a] = r[s, a] + gamma * optimal_policy

		Q = Q_temp		

	return (np.max(Q, axis = 1), np.argmax(Q, axis = 1))

gamma = 0.99
time_steps = 100
optimal_valuation, optimal_policy = find_optimal(P,r, time_steps, gamma)
print ('The optimal valuation is ')
print(optimal_valuation)
print('The optimal policy for stage 0 is')
print(optimal_policy[0])
print('The optimal policy for stage 1 is')
print(optimal_policy[1])