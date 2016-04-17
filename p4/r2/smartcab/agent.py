from __future__ import print_function
import pandas as pd
from multiprocessing.dummy import Pool
import time
import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from collections import namedtuple


class Actions:
    NONE = None
    FORWARD = 'forward'
    LEFT = 'left'
    RIGHT = 'right'
    ALL = [NONE, FORWARD, LEFT, RIGHT]

TrafficRule = namedtuple('TrafficRule', ['OK', 'BLOCKED', 'ALL'])
Forward = TrafficRule('F', 'xF', ['F', 'xF'])
Left = TrafficRule('L', 'xL', ['L', 'xL'])
Right = TrafficRule('R', 'xR', ['R', 'xR'])


class WaypointDir:
    NONE = "wX"
    FORWARD = 'wF'
    LEFT = 'wL'
    RIGHT = 'wR'
    ALL = [NONE, FORWARD, LEFT, RIGHT]

    @staticmethod
    def from_world(waypoint_in_world):
        map_dict = dict(zip(Actions.ALL, WaypointDir.ALL))
        return map_dict[waypoint_in_world]


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
        #  deadline = self.env.get_deadline(self)
        F = Forward.OK if inputs["light"] == "green" else Forward.BLOCKED
        L = Left.OK if inputs["light"] == "green" and\
            inputs['oncoming'] != "forward" else Left.BLOCKED
        R = Right.OK if inputs["light"] == "green" or\
            inputs['oncoming'] != "left" and inputs['left'] != "forward"\
            else Right.BLOCKED

        state = self.lv.create_state(
            WaypointDir.from_world(self.next_waypoint),
            L,
            R,
            F
        )
        return state

    def update(self, t):
        # ***********************************
        # .
        # Get the new state from the inputs
        s = self.get_state()

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
        # Collect data for stats checking
        self.lv.record_result(r)

        # ***********************************
        # .
        # Update our q-values with our new state
        s_prime = self.get_state()
        self.lv.update_q_value(s, a, r, s_prime)


class LearningVariables:
    def __init__(self,
                 gamma,
                 learning_rate,
                 random_val,
                 n_trials,
                 q_value_init,
                 show_q_matrix=False,
                 show_q_update=False,
                 show_random_action=False,
                 show_step_info=True):
        self.gamma = round(gamma, 5)
        self.learning_rate = round(learning_rate, 5)
        self.n_trials = n_trials
        self.q_value_init = q_value_init
        self.random_val = random_val
        self.result = 0
        self.penalties = 0
        self.q_values = None
        self.show_q_update = show_q_update
        self.show_q_matrix = show_q_matrix
        self.show_random_action = show_random_action
        self.show_step_info = show_step_info

        # Variables for the trial
    def reset(self):
        self.trial = 0
        self.q_values = {}

    def new_trial(self):
        self.step = 1
        self.trial += 1
        self.trial_rewards = []

    def do_random_action(self):
        """Return True if we should do a random action"""
        if self.trial > self.n_trials * .7:
            return False
        return random.random() < self.random_val

    def record_result(self, reward):
        # Record rewards when we get to the destination
        self.trial_rewards.append(reward)

        if reward > 9:  # Have we hit the destination?
            self.result += 1
        if reward < 0:  # Do we have penalties
            self.penalties += 1
        self.step += 1

    def create_state(self, waypoint, l, r, f, init_q_value=True):
        state = (waypoint, l, r, f)

        if init_q_value and state not in self.q_values:
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
        if self.do_random_action():
            if self.show_random_action:
                print("Random action!")
            actions_to_choose_from = Actions.ALL
        else:
            max_q, actions_to_choose_from = self.get_best_action(state)

        # Choose an action from the list of actions
        action_to_do = random.choice(actions_to_choose_from)
        return action_to_do

    def get_max_q_value_s_prime(self, state):
        # Here we find the max q_value of the state for s_prime
        # NOTE: We should search across all states that have non deterministic
        #       properties, such as l, r and f as these are be completely
        #       random after a transition.

        states_to_search = [self.create_state(state[0], l, r, f, False)
                            for l in Left.ALL
                            for r in Right.ALL
                            for f in Forward.ALL]

        values = [self.q_values[s][a]
                  for s in states_to_search
                  for a in Actions.ALL
                  if s in self.q_values and a in self.q_values[s]]

        # We might have an empty state space
        if not values:
            return 0
        return max(values)

    def update_q_value(self, s, a, r, s_prime):
        self.print_q_matrix((s, a, r, s_prime))

        # Speed enhancement
        max_s_prime = self.get_max_q_value_s_prime(s_prime)
        s_utility = r +\
            (self.gamma * max_s_prime)

        # Show the existing value
        if self.show_q_update:
            print("Updating q_value")
            print("lr:%.3f, r:%s, g:%.3f, max_s_prime:%.3f" %
                  (self.learning_rate,
                   r,
                   self.gamma,
                   max_s_prime))

            # print("Learning_variables:%s, max_s_prime:%s, s_utility:%s"\)
            #       % (self, max_s_prime, s_utility)
            self._print_q_value_line(s)

        # Adjust q(s,a) towards the utility using the learning rate
        self.q_values[s][a] =\
            ((1 - self.learning_rate) * self.q_values[s][a]) +\
            (self.learning_rate * s_utility)

        # Show the updated value
        if self.show_q_update:
            self._print_q_value_line(s)

    def _print_q_value_line(self, s):
        max_q, best_actions = self.get_best_action(s)
        values = ["%s:%+3.3f" % (a, self.q_values[s][a])
                  for a in Actions.ALL]
        print("%30s, %70s, %s" % (s, values, best_actions))

    def print_q_matrix(self, state=None, show_q_matrix=False):
        if self.show_step_info or show_q_matrix:
            print("-" * 150)
            if state is not None:
                print("Details s:%s a:%s: r:%s s':%s" % state)
            print("#Trials:%s, Trial no:%s, Step:%s, Completions:%s" % (
                self.n_trials, self.trial, self.step, self.result))
            print("Trial rewards:%s" % self.trial_rewards)
            if self.show_q_matrix or show_q_matrix:
                print("%30s, %70s, %s" % ("State|", "Q-values |", "Policy"))
                for s in sorted(self.q_values):
                    self._print_q_value_line(s)
                print

    def __str__(self):
        result = "g:%+3.3f, lr:%+3.3f, rv:%+3.3f, qi:%+3.3f, dest:%s, p:%s" %\
            (self.gamma,
             self.learning_rate,
             self.random_val,
             self.q_value_init,
             self.result,
             self.penalties)
        return result

# Global variable to keep track of the number of thread finished
global finished_count
finished_count = 0


def run_test(learning_variables):
    start_time = time.time()

    e = Environment()
    a = e.create_agent(LearningAgent, learning_variables)
    e.set_primary_agent(a, enforce_deadline=True)

    # Run the simulator
    sim = Simulator(e, update_delay=0.0)
    sim.run(n_trials=learning_variables.n_trials)

    global finished_count
    finished_count += 1

    print("Complete:%s, t:%.3f, #finished:%s" %
          (learning_variables, time.time() - start_time, finished_count))
    return learning_variables


def run_threaded(n_samples, n_trials=100):
    """Run the agent for a finite number of trials."""

    learning_variables = []
    for count in range(n_samples):
        lr = LearningVariables(
            gamma=random.random(),                      # (0 < 1)
            learning_rate=random.random(),              # (0 < 1)
            random_val=random.random(),                 # (0 < 1)
            n_trials=n_trials,
            q_value_init=(random.random() - 0.5) * 10,  # (-5 <= 5)
            show_step_info=False)
        learning_variables.append(lr)

    global thead_count
    thead_count = len(learning_variables)

    p = Pool()
    results = p.map(run_test, learning_variables)
    p.close()
    p.join()
    print("Finished %s samples" % n_samples)

    all_results = [(a.gamma,
                    a.learning_rate,
                    a.random_val,
                    a.q_value_init,
                    a.result,
                    a.penalties)
                   for a in results]

    results_filename = 'results_t%s_s%s_%s.csv' %\
        (n_trials,
         n_samples,
         time.strftime("%Y%m%d-%H%M%S"))
    print("Writing results:%s" % results_filename)
    df = pd.DataFrame(all_results, columns=["g", "lr", "rv", "qi", "r", "p"])
    df.to_csv(results_filename, index=False)
    print(df.describe())


def run():
    """Run the agent for a finite number of trials."""

    learning_variables = LearningVariables(
        gamma=0.31003,
        learning_rate=0.52519,
        random_val=0.10123,
        n_trials=100,
        q_value_init=2.47571,
        show_step_info=False
        )

    result = run_test(learning_variables)
    result.print_q_matrix(show_q_matrix=True)

if __name__ == '__main__':
    run()
    # while True:
    #     run_threaded(n_samples=1000)
