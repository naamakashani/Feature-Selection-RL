import torch.nn
from typing import List, Tuple
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from env import *
from RL.extra.agent_mse import *
from ReplayMemory import *
from PrioritiziedReplayMemory import *

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--save_dir",
                    type=str,
                    default='ddqn_models_replay_memory',
                    help="Directory for saved models")
parser.add_argument("--save_guesser_dir",
                    type=str,
                    default='model_guesser',
                    help="Directory for saved guesser model")
parser.add_argument("--directory",
                    type=str,
                    default="C:\\Users\\kashann\\PycharmProjects\\choiceMira\\RL",
                    help="Directory for saved models")
parser.add_argument("--gamma",
                    type=float,
                    default=0.9,
                    help="Discount rate for Q_target")
parser.add_argument("--n_update_target_dqn",
                    type=int,
                    default=10,
                    help="Number of episodes between updates of target dqn")
parser.add_argument("--val_trials_wo_im",
                    type=int,
                    default=15,
                    help="Number of validation trials without improvement")
parser.add_argument("--ep_per_trainee",
                    type=int,
                    default=100000,
                    help="Switch between training dqn and guesser every this # of episodes")
parser.add_argument("--batch_size",
                    type=int,
                    default=128,
                    help="Mini-batch size")
parser.add_argument("--hidden-dim",
                    type=int,
                    default=64,
                    help="Hidden dimension")
parser.add_argument("--capacity",
                    type=int,
                    default=1000000,
                    help="Replay memory capacity")
parser.add_argument("--max-episode",
                    type=int,
                    default=2000,
                    help="e-Greedy target episode (eps will be the lowest at this episode)")
parser.add_argument("--min_epsilon",
                    type=float,
                    default=0.01,
                    help="Min epsilon")
parser.add_argument("--initial_epsilon",
                    type=float,
                    default=1,
                    help="init epsilon")
parser.add_argument("--anneal_steps",
                    type=float,
                    default=1000,
                    help="anneal_steps")
parser.add_argument("--lr",
                    type=float,
                    default=1e-4,
                    help="Learning rate")
parser.add_argument("--weight_decay",
                    type=float,
                    default=0e-4,
                    help="l_2 weight penalty")
parser.add_argument("--lr_decay_factor",
                    type=float,
                    default=0.1,
                    help="LR decay factor")
parser.add_argument("--val_interval",
                    type=int,
                    default=20,
                    help="Interval for calculating validation reward and saving model")


FLAGS = parser.parse_args(args=[])

# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_helper(agent: Agent,
                 minibatch: List[Transition],
                 gamma: float) -> float:
    """Prepare minibatch and train them
    Args:
        agent (Agent): Agent has `train(Q_pred, Q_true)` method
        minibatch (List[Transition]): Minibatch of `Transition`
        gamma (float): Discount rate of Q_target
    Returns:
        float: Loss value
    """
    states = np.vstack([x.state for x in minibatch])
    actions = np.array([x.action for x in minibatch])
    rewards = np.array([x.reward for x in minibatch])
    next_states = np.vstack([x.next_state for x in minibatch])
    done = np.array([x.done for x in minibatch])

    Q_predict = agent.get_Q(states)
    Q_target = Q_predict.clone().cpu().data.numpy()
    max_actions = np.argmax(agent.get_Q(next_states).cpu().data.numpy(), axis=1)
    Q_target[np.arange(len(Q_target)), actions] = rewards + gamma * agent.get_target_Q(next_states)[
        np.arange(len(Q_target)), max_actions].data.numpy() * ~done
    Q_target = agent._to_variable(Q_target).to(device=device)
    return agent.train(Q_predict, Q_target)


def calculate_td_error(state, action, reward, next_state, done, agent, gamma):
    # Current Q-value estimate
    current_q_value = agent.get_Q(state).squeeze()[action]
    if done:
        target_q_value = reward
    else:
        next_q_values = agent.get_target_Q(next_state).squeeze()
        max_next_q_value = max(next_q_values)
        target_q_value = reward + gamma * max_next_q_value

    # TD error
    td_error = target_q_value - current_q_value
    return td_error.item()


def play_episode(env,
                 agent: Agent,
                 replay_memory: ReplayMemory, priorityRM: PrioritizedReplayMemory,
                 eps: float,
                 batch_size: int,
                 train_guesser=True,
                 train_dqn=True, mode='training') -> int:
    """Play an epsiode and train
    Args:
        env (gym.Env): gym environment (CartPole-v0)
        agent (Agent): agent will train and get action
        replay_memory (ReplayMemory): trajectory is saved here
        eps (float): 𝜺-greedy for exploration
        batch_size (int): batch size
    Returns:
        int: reward earned in this episode
    """

    s = env.reset(train_guesser=train_guesser)
    done = False
    total_reward = 0
    mask = env.reset_mask()
    t = 0
    while not done and t < agent.input_dim / 5:
        a = agent.get_action(s, env, eps, mask, mode)
        next_state, r, done, info = env.step(a, mask)
        # if r < 0:
        # done = True
        mask[a] = 0
        total_reward += r
        td = calculate_td_error(s, a, r, next_state, done, agent, FLAGS.gamma)
        priorityRM.push(s, a, r, next_state, done, td)
        replay_memory.push(s, a, r, next_state, done)
        if len(replay_memory) > batch_size:
            if train_dqn:
                minibatch = replay_memory.pop(batch_size)
                train_helper(agent, minibatch, FLAGS.gamma)
                agent.update_learning_rate()
        # if len(priorityRM) > batch_size:
        #     if train_dqn:
        #         minibatch, indices, weights = priorityRM.pop(batch_size)
        #         td_errors = []
        #         for transition, weight in zip(minibatch, weights):
        #             state, action, reward, next_state, done = transition
        #             td_error = calculate_td_error(state, action, reward, next_state, done, agent, FLAGS.gamma)
        #             td_errors.append(td_error)
        #         priorityRM.update_priorities(indices, td_errors)
        #         train_helper(agent, minibatch, FLAGS.gamma)
        #         agent.update_learning_rate()

        t += 1
    return total_reward,t


def get_env_dim(env) -> Tuple[int, int]:
    """Returns input_dim & output_dim
    Args:
        env (gym.Env): gym Environment
    Returns:
        int: input_dim
        int: output_dim
    """
    input_dim = env.guesser.features_size
    output_dim = env.guesser.features_size + 1

    return input_dim, output_dim


def epsilon_annealing(initial_epsilon, min_epsilon, anneal_steps, current_step):
    """
    Epsilon annealing function for epsilon-greedy exploration in reinforcement learning.

    Parameters:
    - initial_epsilon: Initial exploration rate
    - min_epsilon: Minimum exploration rate
    - anneal_steps: Number of steps over which to anneal epsilon
    - current_step: Current step in the learning process

    Returns:
    - epsilon: Annealed exploration rate for the current step
    """
    epsilon = max(min_epsilon, initial_epsilon - (initial_epsilon - min_epsilon) * current_step / anneal_steps)
    return epsilon


# def epsilon_annealing(episode: int, max_episode: int, min_eps: float) -> float:
#     """Returns 𝜺-greedy
#     1.0---|\
#           | \
#           |  \
#     min_e +---+------->
#               |
#               max_episode
#     Args:
#         epsiode (int): Current episode (0<= episode)
#         max_episode (int): After max episode, 𝜺 will be `min_eps`
#         min_eps (float): 𝜺 will never go below this value
#     Returns:
#         float: 𝜺 value
#     """
#
#     slope = (min_eps - 1.0) / max_episode
#     return max(slope * episode + 1.0, min_eps)


def save_networks(i_episode: int, env, agent,
                  val_acc=None) -> None:
    """ A method to save parameters of guesser and dqn """
    if not os.path.exists(FLAGS.save_dir):
        os.makedirs(FLAGS.save_dir)

    if i_episode == 'best':
        guesser_filename = 'best_guesser.pth'
        dqn_filename = 'best_dqn.pth'
    else:
        guesser_filename = '{}_{}_{:1.3f}.pth'.format(i_episode, 'guesser', val_acc)
        dqn_filename = '{}_{}_{:1.3f}.pth'.format(i_episode, 'dqn', val_acc)

    guesser_save_path = os.path.join(FLAGS.save_dir, guesser_filename)
    dqn_save_path = os.path.join(FLAGS.save_dir, dqn_filename)

    # save guesser
    if os.path.exists(guesser_save_path):
        os.remove(guesser_save_path)
    torch.save(env.guesser.cpu().state_dict(), guesser_save_path + '~')
    env.guesser.to(device=device)
    os.rename(guesser_save_path + '~', guesser_save_path)

    # save dqn
    if os.path.exists(dqn_save_path):
        os.remove(dqn_save_path)
    torch.save(agent.dqn.cpu().state_dict(), dqn_save_path + '~')
    agent.dqn.to(device=device)
    os.rename(dqn_save_path + '~', dqn_save_path)


# Function to extract states from replay memory
# def extract_states_from_replay_memory(replay_memory):
#     states = []
#     for experience in replay_memory:
#         state = experience['state']  # Assuming 'state' key holds the state information
#         states.append(state)
#     return np.array(states)


def load_networks(i_episode: int, env, input_dim=26, output_dim=14,
                  val_acc=None) -> None:
    """ A method to load parameters of guesser and dqn """
    if i_episode == 'best':
        guesser_filename = 'best_guesser.pth'
        dqn_filename = 'best_dqn.pth'
    else:
        guesser_filename = '{}_{}_{:1.3f}.pth'.format(i_episode, 'guesser', val_acc)
        dqn_filename = '{}_{}_{:1.3f}.pth'.format(i_episode, 'dqn', val_acc)

    guesser_load_path = os.path.join(FLAGS.save_guesser_dir, guesser_filename)
    dqn_load_path = os.path.join(FLAGS.save_dir, dqn_filename)

    # load guesser
    guesser = Guesser()
    guesser_state_dict = torch.load(guesser_load_path)
    guesser.load_state_dict(guesser_state_dict)
    guesser.to(device=device)

    # load sqn
    dqn = DQN(input_dim, output_dim, FLAGS.hidden_dim)
    dqn_state_dict = torch.load(dqn_load_path)
    dqn.load_state_dict(dqn_state_dict)
    dqn.to(device=device)
    return guesser, dqn


def save_plot_acuuracy_epoch(accuracy_list):
    '''
    Save plot of accuracy per epoch
    :param accuracy_list: list of accuracies per epoch
    '''
    plt.plot(accuracy_list)
    plt.title('Accuracy per validation epoch')
    plt.ylabel('Accuracy')
    plt.xlabel('validation epoch')
    plt.savefig('accuracy_per_validation_epoch_RM_random reward.png')
    plt.show()

def save_plot_reward_epoch(reward_list):
    '''
    Save plot of accuracy per epoch
    :param accuracy_list: list of accuracies per epoch
    '''
    plt.plot(reward_list)
    plt.title('reward per epoch')
    plt.ylabel('reward')
    plt.xlabel('epoch')
    plt.savefig('reward_per_epoch_RM.png')
    plt.show()

def save_plot_step_epoch(steps):
    '''
    Save plot of accuracy per epoch
    :param accuracy_list: list of accuracies per epoch
    '''
    #calc the average of steps list
    print(np.mean(steps))
    plt.plot(steps)
    plt.title('reward per epoch')
    plt.ylabel('reward')
    plt.xlabel('epoch')
    plt.savefig('steps_per_epoch_RM.png')
    plt.show()


def test(env, agent, input_dim, output_dim):
    total_steps = 0
    """ Computes performance nad test data """

    print('Loading best networks')
    env.guesser, agent.dqn = load_networks(i_episode='best', env=env, input_dim=input_dim, output_dim=output_dim)
    y_hat_test = np.zeros(len(env.y_test))
    print('Computing predictions of test data')
    n_test = len(env.X_test)
    for i in range(n_test):
        number_of_steps = 0
        state = env.reset(mode='test',
                          patient=i,
                          train_guesser=False)
        mask = env.reset_mask()
        t = 0
        done = False
        while t < agent.input_dim / 5 and not done:
            number_of_steps += 1
            # select action from policy
            if t == 0:
                action = agent.get_action_not_guess(state, env, eps=0, mask=mask, mode='test')
            else:
                action = agent.get_action(state, env, eps=0, mask=mask, mode='test')
            mask[action] = 0
            # take the action
            state, reward, done, guess = env.step(action, mask, mode='test')

            if guess != -1:
                y_hat_test[i] = env.guess
            t += 1

        if guess == -1:
            a = agent.output_dim - 1
            s2, r, done, info = env.step(a, mask)
            y_hat_test[i] = env.guess
            total_steps += number_of_steps

    C = confusion_matrix(env.y_test, y_hat_test)
    print('confusion matrix: ')
    print(C)
    acc = np.sum(np.diag(C)) / len(env.y_test)
    print('Test accuracy: ', np.round(acc, 3))
    print('Average number of steps: ', np.round(total_steps / n_test, 3))
    return acc


#
def show_sample_paths(n_patients, env, agent):
    """A method to run episodes on randomly chosen positive and negative test patients, and print trajectories to console  """

    # load best performing networks
    print('Loading best networks')
    input_dim, output_dim = get_env_dim(env)
    env.guesser, agent.dqn = load_networks(i_episode='best', env=env, input_dim=input_dim, output_dim=output_dim)

    for i in range(n_patients):
        print('Starting new episode with a new test patient')
        if i % 2 == 0:
            idx = np.random.choice(np.where(env.y_test == 1)[0])
        else:
            idx = np.random.choice(np.where(env.y_test == 0)[0])
        state = env.reset(mode='test',
                          patient=idx,
                          train_guesser=False)

        mask = env.reset_mask()

        # run episode
        for t in range(int(agent.input_dim / 5)):

            # select action from policy
            action = agent.get_action(state, env, eps=0, mask=mask, mode='test')
            mask[action] = 0

            if action != env.guesser.features_size:
                print('Step: {}, Question: '.format(t + 1), env.guesser.question_names[action], ', Answer: ',
                      env.X_test[idx, action])

            # take the action
            state, reward, done, guess = env.step(action, mask, mode='test')

            if guess != -1:
                print('Step: {}, Ready to make a guess: Prob({})={:1.3f}, Guess: y={}, Ground truth: {}'.format(t + 1,
                                                                                                                guess,
                                                                                                                env.probs[
                                                                                                                    guess],
                                                                                                                guess,
                                                                                                                env.y_test[
                                                                                                                    idx]))

                break

        if guess == -1:
            state, reward, done, guess = env.step(agent.output_dim - 1, mask, mode='test')

            print('Step: {}, Ready to make a guess: Prob({})={:1.3f}, Guess: y={}, Ground truth: {}'.format(t + 1,
                                                                                                            guess,
                                                                                                            env.probs[
                                                                                                                guess],
                                                                                                            guess,
                                                                                                            env.y_test[
                                                                                                                idx]))
        print('Episode terminated\n')


def val(i_episode: int,
        best_val_acc: float, env, agent) -> float:
    """ Compute performance on validation set and save current models """

    print('Running validation')
    y_hat_val = np.zeros(len(env.y_val))

    for i in range(len(env.X_val)):
        state = env.reset(mode='val',
                          patient=i,
                          train_guesser=False)
        mask = env.reset_mask()
        t = 0
        done = False
        while t < agent.input_dim / 5 and not done:
            # select action from policy
            if t == 0:
                action = agent.get_action_not_guess(state, env, eps=0, mask=mask, mode='val')

            else:
                action = agent.get_action(state, env, eps=0, mask=mask, mode='val')

            mask[action] = 0
            # take the action
            state, reward, done, guess = env.step(action, mask, mode='val')
            if guess != -1:
                y_hat_val[i] = guess
            t += 1

        if guess == -1:
            a = agent.output_dim - 1
            s2, r, done, info = env.step(a, mask)
            y_hat_val[i] = env.guess

    confmat = confusion_matrix(env.y_val, y_hat_val)
    acc = np.sum(np.diag(confmat)) / len(env.y_val)
    print('Validation accuracy: {:1.3f}'.format(acc))

    if acc >= best_val_acc:
        print('New best acc acheievd, saving best model')
        save_networks(i_episode, env, agent, acc)
        save_networks(i_episode='best', env=env, agent=agent)
    return acc


def main():
    # define environment and agent (needed for main and test)
    env = myEnv(flags=FLAGS,
                device=device)
    input_dim, output_dim = get_env_dim(env)
    agent = Agent(input_dim,
                  output_dim,
                  FLAGS.hidden_dim, FLAGS.lr, FLAGS.weight_decay)

    agent.dqn.to(device=device)
    env.guesser.to(device=device)

    # store best result
    best_val_acc = 0
    val_list = []

    # counter of validation trials with no improvement, to determine when to stop training
    val_trials_without_improvement = 0
    replay_memory = ReplayMemory(FLAGS.capacity)
    priorityRP = PrioritizedReplayMemory(FLAGS.capacity)
    train_dqn = True
    train_guesser = False
    i = 0
    rewards_list = []
    steps=[]
    while val_trials_without_improvement < FLAGS.val_trials_wo_im:
        # if i % (2 * FLAGS.ep_per_trainee) == 0:
        #     train_dqn = False
        #     train_guesser = True
        # if i % (2 * FLAGS.ep_per_trainee) == FLAGS.ep_per_trainee:
        #     train_dqn = True
        #     train_guesser = False

        eps = epsilon_annealing(FLAGS.initial_epsilon, FLAGS.min_epsilon, FLAGS.anneal_steps, i)
        # play an episode
        reward,t = play_episode(env,
                              agent,
                              replay_memory, priorityRP,
                              eps,
                              FLAGS.batch_size,
                              train_dqn=train_dqn,
                              train_guesser=train_guesser, mode='training')
        rewards_list.append(reward)
        steps.append(t)
        if i % FLAGS.val_interval == 0:
            # compute performance on validation set
            new_best_val_acc = val(i_episode=i,
                                   best_val_acc=best_val_acc, env=env, agent=agent)
            val_list.append(new_best_val_acc)

            # update best result on validation set and counter
            if new_best_val_acc > best_val_acc:
                best_val_acc = new_best_val_acc
                val_trials_without_improvement = 0
            else:
                val_trials_without_improvement += 1

        if i % FLAGS.n_update_target_dqn == 0:
            agent.update_target_dqn()
        i += 1

    acc = test(env, agent, input_dim, output_dim)
    steps = np.mean(steps)
    return acc, steps, i

    # save_plot_acuuracy_epoch(val_list)
    # save_plot_reward_epoch(rewards_list)
    # check the avergae number of steps

    # save_plot_step_epoch(steps)
    # show_sample_paths(6, env, agent)


if __name__ == '__main__':
    os.chdir(FLAGS.directory)
    acc_sum = 0
    steps_sum = 0
    epochs_sum = 0
    for i in range(10):
        acc, steps, epochs = main()
        acc_sum += acc
        steps_sum += steps
        epochs_sum += epochs
    print('average accuracy: ', acc_sum / 10)
    print('average steps: ', steps_sum / 10)
    print('average epochs: ', epochs_sum / 10)