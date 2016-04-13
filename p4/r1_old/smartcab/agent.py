import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy as np

import matplotlib.pyplot as plt

class Actions:
    stay = 'stay'
    forward = 'forward'
    left = 'left'
    right = 'right'
    all = (stay, forward, left, right)


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env, learning_variables):
        """
        :param random_action_ratio: Do a random action if a random number is below this
        """
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        self.learning_variables = learning_variables

        # TODO: Initialize any additional variables here
        self.q_values = {}
        self.t = 1
        self.trial = 0

    def reset(self, destination=None):
        self.planner.route_to(destination)
        self.trial += 1

    # def create_state(self,
    #                  # oncoming_car_forward,
    #                  # left_car_forward,
    #                  green_light,
    #                  destination_direction):
    def create_state(self,
                     # oncoming_car_forward,
                     # left_car_forward,
                     l, # Traffic light
                     d): # Direction
                     #t): # Deadline    
        result = locals()
        del result['self']
        result = str(result)
        if not result in self.q_values:
            self.init_q_value_state(result)
        return result

    def init_q_value_state(self, state):
        self.q_values[state] = {}
        for a in Actions.all:
            # Randomise q_values when new to enhance exploration
            self.q_values[state][a] = (random.random()) * 2

    def get_state(self):
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        state = self.create_state(
            # inputs['oncoming'] == 'forward',
            # inputs['left'] == 'forward',
            inputs["light"] == "green",
            self.planner.next_waypoint() #,
            #deadline < 5
        )

        if state not in self.q_values:
            # This is a new state. Let's initialise the q matrix
            self.init_q_value_state(state)
        return state

    def get_max_action(self, state):
        max_q = None
        best_actions = []
        for action in Actions.all:
            q_value = self.q_values[state][action]
            if q_value == max_q:
                # Might have the same q value
                best_actions.append(action)
            elif q_value > max_q:
                # We have a new best q value
                best_actions = [action]
                max_q = q_value
        return max_q, best_actions

    def choose_best_action(self, state):
        # Add some randomness to aid exploration        
        if self.learning_variables.do_random_action(self.trial):
            actions_to_choose_from = Actions.all
        else:
            max_q, actions_to_choose_from = self.get_max_action(state)

        # Choose an action from the list of actions
        action_to_do = random.choice(actions_to_choose_from)
        return action_to_do

    def get_max_q_value_s_prime(self, state):
        max_q = max([self.q_values[self.create_state(light, self.planner.next_waypoint())][action]
                    for light in [True, False]
                    for action in Actions.all])
        return max_q

    def update_q_values(self, s, a, r, s_prime):
        # Speed enhancement
        if self.gamma > 0:
            s_prime_utility = self.get_max_q_value_s_prime(s_prime)
        else:
            s_prime_utility = 0

        s_utility = r +\
            (self.learning_variables.gamma * s_prime_utility)

        # Adjust q(s,a) towards the utility using the learning rate
        learning_rate = self.learning_rate_func(self.trial)
        self.q_values[s][a] =\
            ((1 - learning_rate) * self.q_values[s][a]) +\
            (learning_rate * s_utility)

    def update(self, t):
        # ***********************************
        # .
        # Get the new state from the inputs
        s = self.get_state()

        # ***********************************
        # .
        # Choose an action
        # Find the best action a from q(s,a). (There might be multiple...)
        a = self.choose_best_action(s)        

        # ***********************************
        # .
        # Map our action to the world action
        action_in_world = a
        if action_in_world == Actions.stay:
            action_in_world = None
        # Execute action and get reward
        r = self.env.act(self, action_in_world)

        # ***********************************
        # .
        # Update the q(s,a) <(lr)= r + g * s_prime_utility
        # Calculate utility of the state
        s_prime = self.get_state()
        self.update_q_values(s, a, r, s_prime)

        # ***********************************
        # .
        # Collect data for stats checking
        self.learning_variables.record_result(self.t, r)
        self.t += 1

        # print "State:%s" % s
        # print "Action:%s" % a
        # print "Reward:%s" % r


class VariableDecay:
    """A class to return a value between start and end based on xx of x_steps

    Keyword Arguments
    start -- range start
    end -- range end
    power -- exponential decay. 1 is uniform, 2 is quadratic decay (faster)
    """
    def __init__(self, start, end, power=1):
        self.start = start
        self.end = end
        self.power = power

    def f(self, rate, x):
        result = self.start * ((1.-rate) ** x)
        if result < 0.:
            return 0.
        return result

    def step(self, step, total_steps):
        # Calculate a degredation value
        v = 1.0 * step / total_steps
        drop = (1 - v) ** self.power
        diff = self.end - self.start
        value = drop * diff

        result = self.end - value
        return round(result, 5)

    def __str__(self):
        result = [round(a, 5) for a in [self.start, self.end, self.power]]
        return str(result)


class QLearningVariables:
    def __init__(self, gamma, learning_rate_func, random_action_func, n_trials):
        self.gamma = round(gamma, 5)
        self.learning_rate_func = learning_rate_func
        self.random_action_func = random_action_func
        self.result = 0
        self.n_trials = n_trials

    def learning_rate(self, step):
        """Get the learning rate for step x of n"""
        return self.learning_rate_func.power
        val = self.learning_rate_func.f(0.5, step)
        return val
        val = self.learning_rate_func.step(step, self.n_trials)
        return round(val, 5)

    def do_random_action(self, step):
        """Return True if we should do a random action"""
        val = self.random_action_func.step(step, self.n_trials)
        val = 0.2
        return random.random() < val

    def record_result(self, t, reward):
        if reward >= 10:
            self.result += 1

    def __str__(self):
        result = [str(a) for a in [self.gamma,
                                   self.learning_rate_func,
                                   self.random_action_func,
                                   self.result]]
        return str(result)


import csv
import pandas as pd

# steps = 100
# for a in range(steps):
#     b = [learning_variables.time_adjusted_function(0.5, 0., a, steps, 2)]
#     print b

def range_me(start, end, steps):
    step = 0-(start - end) / (1. * steps)
    return [round(a, 5) for a in np.arange(start, end, step)]


from multiprocessing.dummy import Pool
from multiprocessing import Queue
from threading import Thread


def thread_run(learning_variables):
    #for random_action in [a/10. for a in range(1, 5)]:
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent,
                       learning_variables=learning_variables)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0.0)  # reduce update_delay to speed up simulation
    sim.run(n_trials=learning_variables.n_trials)  # press Esc or close pygame window to quit
    print "Complete:%s" % learning_variables
    return learning_variables


import time
def run():
    """Run the agent for a finite number of trials."""

    start = time.time()

    gamma_range = range_me(-1., 1., 10)
    lr_range = range_me(-1., 1., 10)
    n_trials = 100
    n_times = 5

    lr_power = 0.1
    learning_variables = []
    for gamma in gamma_range:
        for lr_power in lr_range:
                learning_rate_func = VariableDecay(
                    1.0,
                    0.0,
                    lr_power)  # Decay the learning rate over number of trials

                random_action_func = VariableDecay(
                    0.2,
                    0.0,
                    1)  # Decay the random action ofer the number of trials

                for a in range(n_times):
                    lr = QLearningVariables(
                        gamma,
                        learning_rate_func,
                        random_action_func,
                        n_trials)
                    # queue.put(lr)
                    learning_variables.append(lr)

    p = Pool()
    results = p.map(thread_run, learning_variables)
    p.close()
    p.join()
    print len(results)
    print "Finished"

    all_results = [(a.gamma,
                    a.learning_rate_func.power,
                    a.result) for a in results]
    print all_results
    print

    with open('results20.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(all_results)

    df = pd.DataFrame(all_results, columns=["g", "lr", "res"])
    print df.head()
    show(df)

import seaborn as sns    
def show(df):
    grouped = df.groupby(["g", "lr"])
    df1 = grouped.sum().reset_index()[["g", "lr", "res"]]
    df2 = df1.pivot(index='g', columns='lr', values='res')
    sns.heatmap(df2, annot=True, fmt="d")
    plt.show()

if __name__ == '__main__':
    run()
