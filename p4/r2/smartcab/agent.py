
import pandas as pd
from multiprocessing.dummy import Pool
import seaborn as sns
import csv
import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy as np

import matplotlib.pyplot as plt


class Actions:
    NONE = None
    FORWARD = 'forward'
    LEFT = 'left'
    RIGHT = 'right'
    ALL = [NONE, FORWARD, LEFT, RIGHT]


class Light:
    GREEN = 'G'
    RED = 'R'
    ALL = [GREEN, RED]


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env, learning_variables):
        """
        :param random_action_ratio: Do a random action if a random
        number is below this
        """
        # sets self.env = env, state = None, next_waypoint = None,
        # and a default color
        super(LearningAgent, self).__init__(env)
        self.color = 'red'  # override color

        # simple route planner to get next_waypoint
        self.planner = RoutePlanner(self.env, self)

        # Store the learning variables
        self.lv = learning_variables
        self.lv.reset()

    def reset(self, destination=None):
        self.planner.route_to(destination)

        self.last_values = None
        self.lv.new_trial()

    def get_state(self):
        inputs = self.env.sense(self)
        self.next_waypoint = self.planner.next_waypoint()
        deadline = self.env.get_deadline(self)

        state = self.lv.create_state(
            Light.GREEN if inputs["light"] == "green" else Light.RED,
            self.next_waypoint,
            deadline
        )
        return state

    def update(self, t):
        # ***********************************
        # .
        # Get the new state from the inputs
        s = self.get_state()

        # ***********************************
        # .
        # Update q_values if required
        if self.last_values is not None:
            s_last, a_last, r_last = self.last_values
            self.lv.update_q_value(s_last, a_last, r_last, s)

        # ***********************************
        # .
        # Choose an action
        # Find the best action a from q(s,a). (There might be multiple...)
        a = self.lv.choose_action(s)

        # ***********************************
        # .
        # Map our action to the world action
        # Execute action and get reward
        r = self.env.act(self, a)

        # ***********************************
        # .
        # Let's store these values so we can update the q_values when
        # we get into this function again with our new state
        self.last_values = (s, a, r)

        # ***********************************
        # .
        # Collect data for stats checking
        self.lv.record_result(r)


class LearningVariables:
    def __init__(self,
                 gamma,
                 learning_rate,
                 random_val,
                 n_trials,
                 q_value_init):
        self.gamma = round(gamma, 5)
        self.learning_rate = round(learning_rate, 5)
        self.result = 0
        self.n_trials = n_trials
        self.q_value_init = q_value_init
        self.random_val = random_val

        # Variables for the trial
    def reset(self):
        self.trial = 0
        self.q_values = {}
        print "Reset:%s" % str(self)

    def new_trial(self):
        self.step = 1
        self.trial += 1

    def do_random_action(self, step):
        """Return True if we should do a random action"""
        return random.random() < self.random_val

    def record_result(self, reward):
        # Record rewards when we get to the destination
        if reward >= 10:
            self.result += 1
        self.step += 1

    def create_state(self, light, waypoint, deadline):
        state = (light, waypoint, deadline)
        if state not in self.q_values:
            self.init_q_value(state)
        return state

    def init_q_value(self, state):
        self.q_values[state] = {}
        for a in Actions.ALL:
            # Lets create a optimistic world
            self.q_values[state][a] = self.q_value_init

    def get_best_action(self, state):
        """Get the actions with the highest q_value"""
        max_q = None
        best_actions = []
        for action in Actions.ALL:
            q_value = self.q_values[state][action]
            if q_value == max_q:
                # Might have the same q value
                best_actions.append(action)
            elif q_value > max_q:
                # We have a new best q value
                best_actions = [action]
                max_q = q_value
        return max_q, best_actions

    def choose_action(self, state):
        # Add some randomness to aid exploration
        if self.do_random_action(self.trial):
            actions_to_choose_from = Actions.ALL
        else:
            max_q, actions_to_choose_from = self.get_best_action(state)

        # Choose an action from the list of actions
        action_to_do = random.choice(actions_to_choose_from)
        return action_to_do

    def get_max_q_value_s_prime(self, state):
        # Here we choose the max q_values, but we do not include
        # the traffic light, as this can be random
        max_q = max([self.q_values[state][action] for action in Actions.ALL])
        return max_q

    def update_q_value(self, s, a, r, s_prime):
        # Speed enhancement
        s_prime_utility = self.get_max_q_value_s_prime(s_prime)

        s_utility = r +\
            (self.gamma * s_prime_utility)

        # Adjust q(s,a) towards the utility using the learning rate
        self.q_values[s][a] =\
            ((1 - self.learning_rate) * self.q_values[s][a]) +\
            (self.learning_rate * s_utility)

        #print str((s, a, r, s_prime))
        #self.print_info()

    def print_info(self):
        for s in sorted(self.q_values):
            max_q, best_actions = self.get_best_action(s)
            values = ["%s:%+3.3f" % (a, self.q_values[s][a])
                      for a in Actions.ALL]
            print "%40s, %70s, %s" % (s, values, best_actions)
        print

    def __str__(self):
        result = [str(a) for a in [self.gamma,
                                   self.learning_rate,
                                   self.random_val,
                                   self.q_value_init,
                                   self.result]]
        return str(result)


def range_me(start, end, steps):
    step = 0-(start - end) / (1. * steps)
    return [round(a, 5) for a in np.arange(start, end, step)]


def thread_run(learning_variables):
    e = Environment()
    a = e.create_agent(LearningAgent, learning_variables)
    e.set_primary_agent(a, enforce_deadline=True)

    # Run the simulator
    sim = Simulator(e, update_delay=0.0)
    sim.run(n_trials=learning_variables.n_trials)

    print "Complete:%s" % learning_variables
    return learning_variables


def run():
    """Run the agent for a finite number of trials."""
    gamma_range = range_me(0.0, 1.0, 10)
    lr_range = range_me(0.0, 1.0, 10)
    n_trials = 50
    n_times = 2

    print gamma_range
    print lr_range
    learning_variables = []
    for gamma in gamma_range:
        for learning_rate in lr_range:
            for a in range(n_times):
                lr = LearningVariables(
                    gamma=gamma,
                    learning_rate=learning_rate,
                    random_val=0.2,
                    n_trials=n_trials,
                    q_value_init=2.0)
                learning_variables.append(lr)

    p = Pool()
    results = p.map(thread_run, learning_variables)
    p.close()
    p.join()
    print "Finished"

    all_results = [(a.gamma,
                    a.learning_rate,
                    a.result) for a in results]

    with open('results.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(all_results)

    df = pd.DataFrame(all_results, columns=["g", "lr", "res"])
    show(df)


def show(df):
    grouped = df.groupby(["g", "lr"])
    df1 = grouped.sum().reset_index()[["g", "lr", "res"]]
    df2 = df1.pivot(index='g', columns='lr', values='res')
    sns.heatmap(df2, annot=True, fmt="d")
    plt.show()

if __name__ == '__main__':
    run()
