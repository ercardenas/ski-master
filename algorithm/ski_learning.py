


def get_actions_for_env(env):

    # Assume this is a Discrete Space from the Atari environments.
    # TODO figure out how to remove this assumption (if neccesary for
    # project)
    all_actions = list(range(env.action_space.n))
    
    def actions_function(state):
        return all_actions

    return actions_function
    

def ski_ram_base_feature_extractor(observation, action):
    """ Observation is the content of the RAM as an array.

    Simply make features 

    name: position in ram,
    value: it value
    
    """

    features = [('action', action)]
    for i, v in enumerate(observation):
        features.append(("ram[{}]".format(i), v))
    return features
