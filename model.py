import numpy as np
import copy
import gym
import matplotlib.pyplot as plt
import tensorflow as tf
from stable_baselines.common.tf_util import numel
from trpo_mpi import TRPO
from adversary import TransitionClassifier


class GetFlat(object):
    def __init__(self, var_list, sess=None):
        """
        Get the parameters as a flat vector

        :param var_list: ([TensorFlow Tensor]) the variables
        :param sess: (TensorFlow Session)
        """
        print([tf.reshape(v, [numel(v)]) for v in var_list])
        self.operation = tf.concat(axis=0, values=[tf.reshape(v, [numel(v)]) for v in var_list])
        self.sess = sess

    def __call__(self):
        if self.sess is None:
            return tf.get_default_session().run(self.operation)
        else:
            return self.sess.run(self.operation)

class GAIL(TRPO):
    """
    Generative Adversarial Imitation Learning (GAIL)

    .. warning::

        Images are not yet handled properly by the current implementation


    :param policy: (ActorCriticPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, CnnLstmPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param expert_dataset: (ExpertDataset) the dataset manager
    :param gamma: (float) the discount value
    :param timesteps_per_batch: (int) the number of timesteps to run per batch (horizon)
    :param max_kl: (float) the Kullback-Leibler loss threshold
    :param cg_iters: (int) the number of iterations for the conjugate gradient calculation
    :param lam: (float) GAE factor
    :param entcoeff: (float) the weight for the entropy loss
    :param cg_damping: (float) the compute gradient dampening factor
    :param vf_stepsize: (float) the value function stepsize
    :param vf_iters: (int) the value function's number iterations for learning
    :param hidden_size: ([int]) the hidden dimension for the MLP
    :param g_step: (int) number of steps to train policy in each epoch
    :param d_step: (int) number of steps to train discriminator in each epoch
    :param d_stepsize: (float) the reward giver stepsize
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
        WARNING: this logging can take a lot of space quickly
    """

    def __init__(self, policy, env, expert_dataset=None,
                 hidden_size_adversary=100, adversary_entcoeff=1e-3,
                 g_step=3, d_step=1, d_stepsize=3e-4, verbose=0,
                 _init_setup_model=True, **kwargs):
        super().__init__(policy, env, verbose=verbose, _init_setup_model=False, **kwargs)
        self.using_gail = True
        self.expert_dataset = expert_dataset
        self.g_step = g_step
        self.d_step = d_step
        self.d_stepsize = d_stepsize
        self.hidden_size_adversary = hidden_size_adversary
        self.adversary_entcoeff = adversary_entcoeff
        # self.observation_space.dtype = np.float32
        # self.observation_space.shape = (np.prod(self.observation_space.shape),)
        # self.obs_space = copy.deepcopy(self.observation_space)
        # self.obs_space.dtype = np.float32
        # self.obs_space.shape = (np.prod(self.observation_space.shape),)
        # self.reward_giver = TransitionClassifier(self.observation_space, self.action_space,
        #                                                      self.hidden_size_adversary,
        #                                                      entcoeff=self.adversary_entcoeff)
        if _init_setup_model:
            self.setup_model()

    def learn(self, total_timesteps, callback=None, seed=None, log_interval=100, tb_log_name="GAIL",
              reset_num_timesteps=True):
        assert self.expert_dataset is not None, "You must pass an expert dataset to GAIL for training"
        return super().learn(total_timesteps, callback, seed, log_interval, tb_log_name, reset_num_timesteps)

    
    def pretrain(self, dataset, n_epochs=10, learning_rate=0.0001, adam_epsilon=1e-08, val_interval=100):
        """
        Pretrain a model using behavior cloning:
        supervised learning given an expert dataset.

        NOTE: only Box and Discrete spaces are supported for now.

        :param dataset: (ExpertDataset) Dataset manager
        :param n_epochs: (int) Number of iterations on the training set
        :param learning_rate: (float) Learning rate
        :param adam_epsilon: (float) the epsilon value for the adam optimizer
        :param val_interval: (int) Report training and validation losses every n epochs.
            By default, every 10th of the maximum number of epochs.
        :return: (BaseRLModel) the pretrained model
        """
        continuous_actions = isinstance(self.action_space, gym.spaces.Box)
        discrete_actions = isinstance(self.action_space, gym.spaces.Discrete)

        assert discrete_actions or continuous_actions, 'Only Discrete and Box action spaces are supported'

        # Validate the model every 10% of the total number of iteration
        if val_interval is None:
            # Prevent modulo by zero
            if n_epochs < 10:
                val_interval = 1
            else:
                val_interval = int(n_epochs / 10)

        with self.graph.as_default():
            with tf.variable_scope('pretrain'):
                if continuous_actions:
                    obs_ph, actions_ph, deterministic_actions_ph = self._get_pretrain_placeholders()
                    loss = tf.reduce_mean(tf.square(actions_ph - deterministic_actions_ph))
                else:
                    obs_ph, actions_ph, actions_logits_ph = self._get_pretrain_placeholders()
                    # actions_ph has a shape if (n_batch,), we reshape it to (n_batch, 1)
                    # so no additional changes is needed in the dataloader
                    actions_ph = tf.expand_dims(actions_ph, axis=1)
                    one_hot_actions = tf.one_hot(actions_ph, self.action_space.n)
                    loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                        logits=actions_logits_ph,
                        labels=tf.stop_gradient(one_hot_actions)
                    )
                    loss = tf.reduce_mean(loss)
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=adam_epsilon)
                optim_op = optimizer.minimize(loss, var_list=self.params)

            self.sess.run(tf.global_variables_initializer())

        if self.verbose > 0:
            print("Pretraining with Behavior Cloning...")
        
        data = []
        ep = []
        for epoch_idx in range(int(n_epochs)):
            train_loss = 0.0
            # Full pass on the training set
            for _ in range(len(dataset.train_loader)):
                expert_obs, expert_actions = dataset.get_next_batch('train')
                expert_obs = np.array([np.concatenate((i.reshape(self.observation_space.shape[:-1] + (3,)), np.full((90,130,1),255)), axis=2) for i in expert_obs])
                feed_dict = {
                    obs_ph: expert_obs,
                    actions_ph: expert_actions,
                }
                train_loss_, _ = self.sess.run([loss, optim_op], feed_dict)
                train_loss += train_loss_

            train_loss /= len(dataset.train_loader)

            if self.verbose > 0 and (epoch_idx + 1) % val_interval == 0:
                val_loss = 0.0
                # Full pass on the validation set
                for _ in range(len(dataset.val_loader)):
                    expert_obs, expert_actions = dataset.get_next_batch('val')
                    expert_obs = np.array([np.concatenate((i.reshape(self.observation_space.shape[:-1] + (3,)), np.full((90,130,1),255)), axis=2) for i in expert_obs])
                    val_loss_, = self.sess.run([loss], {obs_ph: expert_obs,
                                                        actions_ph: expert_actions})
                    val_loss += val_loss_

                val_loss /= len(dataset.val_loader)
                if self.verbose > 0:
                    print("==== Training progress {:.2f}% ====".format(100 * (epoch_idx + 1) / n_epochs))
                    print('Epoch {}'.format(epoch_idx + 1))
                    print("Training loss: {:.6f}, Validation loss: {:.6f}".format(train_loss, val_loss))
                    print()
                ep.append(epoch_idx)
                data.append([train_loss, val_loss])
            # Free memory
            del expert_obs, expert_actions
        if self.verbose > 0:
            print("Pretraining done.")
        data = np.array(data)
        plt.plot(ep, data[:,0])
        plt.plot(ep, data[:,1])
        plt.legend(["train_loss", "val_loss"])
        plt.savefig("plot.jpg")
        return self