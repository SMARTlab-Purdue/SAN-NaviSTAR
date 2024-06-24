import glob
import os
import numpy as np
import torch
import torch.nn as nn
import numpy


def clip_action(action, low=None, high=None, clip_norm=False, max_norm=None):
    # action [batch, action_dim] or [action_dim]
    if type(action) == numpy.ndarray:
        numpy_flag = True
        action = torch.from_numpy(action)
    elif type(action) == torch.Tensor:
        numpy_flag = False
    if clip_norm:
        action_norm = torch.norm(action, dim=-1, keepdim=True)
        action_norm = action_norm.clamp(min=max_norm)
        action = action / action_norm * max_norm

    else:
        action = action.clamp(min=low, max=high)

    if numpy_flag:
        action = action.numpy()

    return action

def check(input):
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output

def check_reverse(input):
    output = _t2n(input) if type(input) == torch.Tensor else input
    return output

def _t2n(x):
    return x.detach().cpu().numpy()

# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def cleanup_log_dir(log_dir):
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)

def tile_images(img_nhwc):
    """
    Tile N images into one big PxQ image
    (P,Q) are chosen to be as close as possible, and if N
    is square, then P=Q.
    input: img_nhwc, list or array of images, ndim=4 once turned into array
        n = batch index, h = height, w = width, c = channel
    returns:
        bigim_HWc, ndarray with ndim=3
    """
    img_nhwc = np.asarray(img_nhwc)
    N, h, w, c = img_nhwc.shape
    H = int(np.ceil(np.sqrt(N)))
    W = int(np.ceil(float(N)/H))
    img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0]*0 for _ in range(N, H*W)])
    img_HWhwc = img_nhwc.reshape(H, W, h, w, c)
    img_HhWwc = img_HWhwc.transpose(0, 2, 1, 3, 4)
    img_Hh_Ww_c = img_HhWwc.reshape(H*h, W*w, c)
    return img_Hh_Ww_c

def draw(robot_traj, robot_goal, human_traj, status, nav_time=None):
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
    robot_color = 'orange'
    goal_color = 'red'
    human_color = ['yellow', 'green', 'blue', 'purple', 'pink', 'gray', 'brown', 'black', 'chocolate', 'indigo'] * 2
    collision_color = 'red'
    success_color = 'green'
    timeout_color = 'goldenrod'
    fig = Figure(figsize=(10, 10), dpi=100)
    ax = fig.add_subplot()
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    artists = []

    robot_num, traj_len, human_num = robot_traj.shape[0], robot_traj.shape[1], human_traj.shape[0]

    for i in range(robot_num):
        alpha = np.linspace(0.4, 1, traj_len)
        for j in range(traj_len):
            c = plt.Circle(robot_traj[i, j], 0.1, fill=True, color=robot_color, alpha=alpha[j])
            ax.add_artist(c)
        plt.plot(robot_traj[i, :, 0], robot_traj[i, :, 1], color=robot_color)
        goal = mlines.Line2D(robot_goal[i, [0]], robot_goal[i, [1]], color=goal_color, marker='*',
                             linestyle='None', markersize=15, label='Goal')

        ax.add_artist(goal)
        artists.append(goal)

    for i in range(human_num):
        traj_len = len(human_traj[i])
        alpha = np.linspace(0.4, 1, traj_len)
        for j in range(traj_len):
            c = plt.Circle(human_traj[i, j], 0.2, fill=False, color=human_color[i], alpha=alpha[j])
            ax.add_artist(c)

    r = mlines.Line2D([100], [100], color=robot_color, marker='.', linestyle='None',
                      markersize=25, label='Robot')
    g = mlines.Line2D([100], [100], color=goal_color, marker='*', linestyle='None',
                      markersize=15, label='Goal')
    h = mlines.Line2D([100], [100], color='black', marker='o', linestyle='None',
                      markersize=15, markerfacecolor='white', label='Human')
    ax.legend([r, g, h], ['Robot', 'Goal', 'Human'], fontsize=16)


    if status == 'Collision':
        ax.text(0, 9, 'Collision', color=collision_color, fontsize=16)
    elif status == 'Success':
        ax.text(0, 9, 'Success', color=success_color, fontsize=16)
        ax.text(0, 7, 'Nav Time: ' + str(nav_time) + 's', color=success_color, fontsize=16)
    else:
        ax.text(0, 9, 'Timeout', color=timeout_color, fontsize=16)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()
    img = np.frombuffer(s, np.uint8).reshape((height, width, 4))
    plt.close('all')

    return img

