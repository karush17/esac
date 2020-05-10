import numpy as np

class CyclicMDP:
    def __init__(self):
        self.end           = False
        self.current_state = 1
        self.num_actions   = 3
        self.num_states    = 3
        self.p_right       = 0.5

    def reset(self):
        self.end = False
        self.current_state = 1
        state = np.zeros(self.num_states)
        state[self.current_state - 1] = 1.
        return state

    def step(self, action):
        if self.current_state == 1 or self.current_state == 2:
            if action == 1:
                    self.current_state += 1
                    reward = 1
                    
            else:
                reward = -1
                self.end = True

        else:
            if action == 1:
                    self.current_state = 1
                    reward = 2
                    
            else:
                reward = -1
                self.end = True
	    
        state = np.zeros(self.num_states)
        state[self.current_state - 1] = 1
        return state, reward, self.end, {}