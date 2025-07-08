'''
  Utils functions and some configs.
  @python version : 3.6.8
'''

import os, re, copy, time, random, datetime, argparse
import numpy as np
# import tensorflow as tf


nowTime = datetime.datetime.now().strftime('%y-%m-%d%H:%M:%S')
parser = argparse.ArgumentParser(description="Process running arguments")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

# hype parameter for PPO training
hype_parameters = {
    "gamma": 0.99,
    "lamda": 0.95,
    "batch_size": 512,
    "epoch_num": 10,
    "clip_value": 0.2,
    "c_1": 3,
    "c_2": 0.001,
    "init_lr": 3e-4,
    "d_lr": 3e-4,
    "lr_epsilon": 1e-6
}

def generate_xml_path():
    import gymnasium as gym, os
    xml_path = os.path.join(gym.__file__[:-11], 'envs/mujoco/assets')

    assert os.path.exists(xml_path)

    return xml_path


gym_xml_path = generate_xml_path()


def record_data(file, content):
    with open(file, 'a+') as f:
        f.write('{}\n'.format(content))


def check_path(path):
    try:
        if not os.path.exists(path):
            os.mkdir(path)
    except FileExistsError:
        pass

    return path


def update_xml(index, env_name):
    xml_name = parse_xml_name(env_name)
    os.system('cp xml_path/{0}/{1} {2}/{1}}'.format(index, xml_name, gym_xml_path))

    time.sleep(0.2)


def parse_xml_name(env_name):
    if 'walker' in env_name.lower():
        xml_name = "walker2d.xml"
    elif 'hopper' in env_name.lower():
        xml_name = "hopper.xml"
    elif 'halfcheetah' in env_name.lower():
        xml_name = "half_cheetah.xml"
    elif "ant" in env_name.lower():
        xml_name = "ant.xml"
    elif "reacher" in env_name.lower():
        xml_name = "reacher.xml"
    elif "invertedpendulum" in env_name.lower():
        xml_name = "inverted_pendulum.xml"
    else:
        raise RuntimeError("No available environment named \'%s\'" % env_name)

    return xml_name


def update_source_env(env_name):
    xml_name = parse_xml_name(env_name)

    os.system(
        'cp xml_path/source_file/{0} {1}/{0}'.format(xml_name, gym_xml_path))

    time.sleep(0.2)


def update_target_env_gravity(variety_degree, env_name):
    xml_name = parse_xml_name(env_name)

    with open('xml_path/source_file/{}'.format(xml_name), "r+") as f:

        new_f = open('xml_path/target_file/{}'.format(xml_name), "w")
        for line in f.readlines():
            if "gravity" in line:
                pattern = re.compile(r"gravity=\"(.*?)\"")
                a = pattern.findall(line)
                friction_list = a[0].split(" ")
                new_friction_list = []
                for num in friction_list:
                    new_friction_list.append(variety_degree * float(num))

                replace_num = " ".join(str(i) for i in new_friction_list)
                replace_num = "gravity=\"" + replace_num + "\""
                sub_result = re.sub(pattern, str(replace_num), line)

                new_f.write(sub_result)
            else:
                new_f.write(line)

        new_f.close()

    os.system(
        'cp xml_path/target_file/{0} {1}/{0}'.format(xml_name, gym_xml_path))

    time.sleep(0.2)


def update_target_env_density(variety_degree, env_name):
    xml_name = parse_xml_name(env_name)

    with open('xml_path/source_file/{}'.format(xml_name), "r+") as f:

        new_f = open('xml_path/target_file/{}'.format(xml_name), "w")
        for line in f.readlines():
            if "density" in line:
                pattern = re.compile(r'(?<=density=")\d+\.?\d*')
                a = pattern.findall(line)
                current_num = float(a[0])
                replace_num = current_num * variety_degree
                sub_result = re.sub(pattern, str(replace_num), line)

                new_f.write(sub_result)
            else:
                new_f.write(line)

        new_f.close()

    os.system(
        'cp xml_path/target_file/{0} {1}/{0}'.format(xml_name, gym_xml_path))

    time.sleep(0.2)


def update_target_env_friction(variety_degree, env_name):
    xml_name = parse_xml_name(env_name)

    with open('xml_path/source_file/{}'.format(xml_name), "r+") as f:

        new_f = open('xml_path/target_file/{}'.format(xml_name), "w")
        for line in f.readlines():
            if "friction" in line:
                pattern = re.compile(r"friction=\"(.*?)\"")
                a = pattern.findall(line)
                friction_list = a[0].split(" ")
                new_friction_list = []
                for num in friction_list:
                    new_friction_list.append(variety_degree * float(num))

                replace_num = " ".join(str(i) for i in new_friction_list)
                replace_num = "friction=\"" + replace_num + "\""
                sub_result = re.sub(pattern, str(replace_num), line)

                new_f.write(sub_result)
            else:
                new_f.write(line)

        new_f.close()

    os.system(
        'cp xml_path/target_file/{0} {1}/{0}'.format(xml_name, gym_xml_path))

    time.sleep(0.2)


def generate_log(extra=None):
    print(extra)
    record_data('documents/{}/data/log.txt'.format(args.log_index), "{}".format(extra))


def get_gaes(rewards, v_preds, v_preds_next):
    deltas = [r_t + hype_parameters["gamma"] * v_next - v for r_t, v_next, v in zip(rewards, v_preds_next, v_preds)]
    # calculate generative advantage estimator(lambda = 1), see ppo paper eq(11)
    gaes = copy.deepcopy(deltas)
    for t in reversed(range(len(gaes) - 1)):  # is T-1, where T is time step which run policy
        gaes[t] = gaes[t] + hype_parameters["gamma"] * hype_parameters["lamda"] * gaes[t + 1]

    return gaes


def get_return(rewards):
    dis_rewards = np.zeros_like(rewards).astype(np.float32)
    running_add = 0
    for t in reversed(range(len(rewards))):
        running_add = running_add * hype_parameters["gamma"] + rewards[t]
        dis_rewards[t] = running_add

    return dis_rewards


def set_global_seeds(i):
    myseed = i  # + 1000 * rank if i is not None else None
    try:
        import tensorflow as tf
        tf.set_random_seed(myseed)
    except Exception as e:
        print("Check your tensorflow version")
        raise e
    np.random.seed(myseed)
    random.seed(myseed)




def check_file_path():
    check_path("./documents")
    check_path("./result")
    check_path("./result/summary")
    check_path("./documents/%s" % args.log_index)

    
    
import numpy as np

# from https://github.com/joschu/modular_rl
# http://www.johndcook.com/blog/standard_deviation/


class RunningStat(object):
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape


class ZFilter:
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """

    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip

        self.rs = RunningStat(shape)
        self.fix = False

    def __call__(self, x, update=True):
        if update and not self.fix:
            self.rs.push(x)
        if self.demean:
            x = x - self.rs.mean
        if self.destd:
            x = x / (self.rs.std + 1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

########### NEW ##############

class EMAStat(object):
    def __init__(self, shape, alpha=0.1):
        """
        Initializes the EMA statistics.
        
        Args:
            shape (tuple): The shape of the data.
            alpha (float): The update rate for the EMA.
        """
        self.alpha = alpha
        self.mean = np.zeros(shape)
        self.var = np.zeros(shape)
        self.initialized = False

    def update(self, x):
        """
        Update the running mean and variance using an exponential moving average.
        
        Args:
            x (np.array): New sample data.
        """
        x = np.asarray(x)
        if not self.initialized:
            self.mean = x.copy()
            self.var = np.zeros_like(x)
            self.initialized = True
        else:
            # Update EMA for the mean
            self.mean = (1 - self.alpha) * self.mean + self.alpha * x
            # Update EMA for the variance based on the new mean
            self.var = (1 - self.alpha) * self.var + self.alpha * np.square(x - self.mean)

class EMAZFilter:
    """
    Normalizes input data using exponential moving averages for mean and variance.
    
    This filter subtracts the EMA mean and divides by the EMA standard deviation.
    It can also clip the normalized output to a specified range.
    """
    def __init__(self, shape, alpha=0.1, demean=True, destd=True, clip=10.0):
        """
        Args:
            shape (tuple): Shape of the data.
            alpha (float): Update rate for the EMA.
            demean (bool): Whether to subtract the mean.
            destd (bool): Whether to divide by the standard deviation.
            clip (float): Value to clip the normalized output.
        """
        self.demean = demean
        self.destd = destd
        self.clip = clip
        self.ema_stat = EMAStat(shape, alpha=alpha)

    def __call__(self, x, update=True):
        """
        Normalize the input data.
        
        Args:
            x (np.array): The input data.
            update (bool): Whether to update the running statistics with x.
            
        Returns:
            np.array: The normalized data.
        """
        x = np.asarray(x)
        if update:
            self.ema_stat.update(x)
        if self.demean:
            x = x - self.ema_stat.mean
        if self.destd:
            x = x / (np.sqrt(self.ema_stat.var) + 1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x
    
#Replay buffer definition

class ReplayBuffer:
    def __init__(self, memory_size, batch_size):
        """
        Initializes the replay buffer.
        
        Parameters:
        - memory_size: Maximum number of experiences to store.
        - batch_size: Number of experiences to sample in each batch.
        """
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.buffer = []

    def add(self, state, action, reward, next_state, done=0):
        """
        Adds a new experience tuple to the buffer. If the buffer is full,
        the oldest experience is removed.
        
        Parameters:
        - state: The current state.
        - action: The action taken.
        - reward: The reward received.
        - next_state: The subsequent state.
        - done: Indicator if the episode ended (default is 0).
        """
        if len(self.buffer) >= self.memory_size:
            # Remove the oldest experience to maintain memory size
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self):
        """
        Randomly samples a batch of experiences from the buffer.
        Returns a tuple of (states, actions, rewards, next_states, dones),
        each converted to a NumPy array.
        """
        # Ensure we sample only the number of experiences we have (if less than batch_size)
        sample_size = min(self.batch_size, len(self.buffer))
        batch = random.sample(self.buffer, sample_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), 
                np.array(actions), 
                np.array(rewards), 
                np.array(next_states), 
                np.array(dones))
    
    def clear(self):
        """
        Clears the buffer, removing all stored experiences.
        """
        self.buffer = []
        print("Replay buffer cleared.")

    def __len__(self):
        """
        Returns the current number of experiences stored in the buffer.
        """
        return len(self.buffer)
