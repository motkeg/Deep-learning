import tensorflow as tf
import numpy as np

import policy

import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko


class A2CAgent:

    def __init__(self, model):
        self.model = model
        self.params = {'value': 0.5, 'entropy': 0.0001, 'gamma': 0.99}
        self.model.compile(
            optimizer=ko.RMSprop(lr=0.0007),
            # define separate losses for policy logits and value 
            loss=[self._logits_loss, self._value_loss]
        )
        #model.summary()

    def test(self, env, render=True):
        obs, done, ep_reward = env.reset(), False, 0
        while not done:
            action, _ = self.model.action_value(obs[None, :])
            obs, reward, done, _ = env.step(action)
            ep_reward += reward
            if render:
                env.render()
        return ep_reward
    
    
    def train(self, env, batch_sz=32, updates=1000):
        # storage helpers for a single batch of data
        actions = np.empty((batch_sz,), dtype=np.int32)
        rewards, dones, values = np.empty((3, batch_sz))
        obs_shape = env.observation_space.shape
        observations = np.empty((batch_sz,) + obs_shape)
        # collect samples, send to optimizer, repeat updates times
        ep_rews = [0.0]
        next_obs = env.reset()
        for update in range(updates):
            for step in range(batch_sz):
                observations[step] = next_obs.copy()
                a, v = self.model.action_value(next_obs[None, :])
                actions[step], values[step] = a, v
                next_obs, rewards[step], dones[step], _ = env.step(actions[step])
                ep_rews[-1] += rewards[step]
                if dones[step]:
                    ep_rews.append(0.0)
                    next_obs = env.reset()
            _, next_value = self.model.action_value(next_obs[None, :])
            returns, advs = self._returns_advantages(rewards, dones, values, next_value)
            # a trick to input actions and advantages through same API
            acts_and_advs = np.concatenate([actions[:, None], advs[:, None]], axis=-1)
            # performs a full training step on the collected batch
            # note: no need to mess around with gradients, Keras API handles it
            losses = self.model.train_on_batch(observations, [acts_and_advs, returns])
        return ep_rews
    
    def _returns_advantages(self, rewards, dones, values, next_value):
        # next_value is the bootstrap value estimate of a future state (the critic)
        returns = np.append(np.zeros_like(rewards), next_value, axis=-1)
        # returns are calculated as discounted sum of future rewards
        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + self.params['gamma'] * returns[t+1] * (1-dones[t])
        returns = returns[:-1]
        # advantages are returns - baseline, value estimates in our case
        advantages = returns - values
        return returns, advantages

    def _value_loss(self, returns, value):
        # value loss as MSE between value estimates and returns
        return self.params['value']*kls.mean_squared_error(returns, value)
    def _logits_loss(self, acts_and_advs, logits):
        # a trick to input actions and advantages through same API
        actions, advantages = tf.split(acts_and_advs, 2, axis=-1)
        # polymorphic CE loss fn, supports sparse and weighted
        # from_logits argument ensures normalized probabilities
        cross_entropy = kls.CategoricalCrossentropy(from_logits=True)
        # policy loss is defined by policy gradients, weighted by advantages
        # note: we only calculate the loss on the actions we've actually taken
        # thus under the hood a sparse version of CE loss will be executed
        actions = tf.cast(actions, tf.int32)
        policy_loss = cross_entropy(actions, logits, sample_weight=advantages)
        # entropy loss can be calculated via CE over itself
        entropy_loss = cross_entropy(logits, logits)
        # here signs are flipped because optimizer minimizes
        return policy_loss - self.params['entropy']*entropy_loss
    
    
