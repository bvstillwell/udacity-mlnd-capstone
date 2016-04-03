import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator


class Actions:
    none = None
    forward = 'forward'
    left = 'left'
    right = 'right'
    all = (none, forward, left, right)


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        
        # TODO: Initialize any additional variables here
        self.q_values = {}
        self.gamma = 0.4
        self.t = 1
        self.trials = 0

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # Create a random state to start
        self.current_state = self.create_state(
            #random.choice([True, False]),  # oncoming_car_forward
            #random.choice([True, False]),  # left_car_forward
            random.choice([True, False]),  # green_light
            random.choice(Actions.all),  # destination_dir
            )
        self.current_state = None

        if str(self.current_state) not in self.q_values:
            # setup some q_values and choose a random direction
            self.init_q_value_state(self.current_state)

        self.trials += 1

    def create_state(self,
                     #oncoming_car_forward,
                     #left_car_forward,
                     green_light,
                     destination_direction):
        result = locals()
        del result['self']
        return result

    def init_q_value_state(self, state):
        self.q_values[str(state)] = {}
        for a in Actions.all:
            #self.q_values[str(state)][str(a)] = 0.
            self.q_values[str(state)][str(a)] = (random.random() -0.5) * 20
        print "INIT"

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # Get the current state
        new_state = self.create_state(
            #inputs['oncoming'] == 'forward',
            #inputs['left'] == 'forward',
            inputs["light"] == "green",
            self.next_waypoint
            )

        # Choose an action based on best q-value, or random otherwise
        action = None
        if str(new_state) not in self.q_values:
            # setup some q_values and choose a random direction
            self.init_q_value_state(new_state)
            action = random.choice(Actions.all)
            max_q_value = 0.
        else:
            # Choose best move
            max_q_value = None
            actions = []
            for a in self.q_values[str(new_state)]:
                test_val = self.q_values[str(new_state)][a]
                if test_val == max_q_value:
                    actions.append(a)
                elif test_val > max_q_value:
                    actions = [a]
                    max_q_value = test_val

            # Pick an action from the possible ones
            action = random.choice(actions)

        e = random.random()
        if e > .9:
            print "random" * 5
            action = random.choice(Actions.all)

        if action == 'None':
            action = None

        # action = self.next_waypoint
        # if not new_state["green_light"]:
        #     action = None

        # Execute action and get reward
        reward = self.env.act(self, action)

        # Update the q-value of the previous state
        lr = 1. / self.t
        self.t += 1
        new_q_value = reward + (self.gamma * max_q_value)
        old_q_value = self.q_values[str(self.current_state)][str(action)]
        adjusted_q_value = (1 - lr) * old_q_value + lr * new_q_value
        self.q_values[str(self.current_state)][str(action)] = adjusted_q_value

        # Set the current state to the new state
        self.current_state = new_state

        # for a in self.q_values:
        #     # print "%s %s" % (a, self.q_values[a])
        #     print "%s" % (self.q_values[a])
        # print
        #print

        output = str(reward)
        output += ", %s" % str(action)
        output += ", %s" % str(self.next_waypoint)
        output += ", %s" % str(new_state["green_light"])

        output += ", %s" % len(self.q_values)
        if reward < 0:
            print output
        else:
            print str(action)

        print "\n".join(self.q_values)
        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0.0)  # reduce update_delay to speed up simulation
    sim.run(n_trials=20)  # press Esc or close pygame window to quit


if __name__ == '__main__':
    run()
