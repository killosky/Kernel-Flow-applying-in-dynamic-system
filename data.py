

import autograd
import autograd.numpy as np
import scipy
import scipy.integrate
import torch

from args import get_args

def hamiltonian_fn(coords):
    '''
    :param coords: coordinate
    :return: the Hamiltonian of the system
    '''
    q, p = np.split(coords, 2)
    H = 3*(1-np.cos(q)) + p**2    # pendulum hamiltonian
    return H

def dynamics_fn(t, coords):
    '''
    :param t: time
    :param coords: coordinate
    :return:dp/dt, -dq/dt of the system
    '''
    dcoords = autograd.grad(hamiltonian_fn)(coords)
    dqdt, dpdt = np.split(dcoords, 2)
    S = np.concatenate([dpdt, -dqdt], axis=-1)
    return S

def get_trajectory(t_span=[0,3], timescale=15, radius=None, y0=None, noise_std=0.1):
    '''
    :param t_span: time span
    :param timescale: the point number of the integral
    :param radius: the radius of the Pendulum
    :param y0: the initial position of the Pendulum
    :param noise_std: noise parameter
    :param kwargs:
    :return: trajectory of the system
    '''
    t_eval = np.linspace(t_span[0], t_span[1], int(timescale * (t_span[1] - t_span[0])))

    # get initial state
    if y0 is None:
        y0 = np.random.rand(2)*2.-1
    if radius is None:
        radius = np.random.rand() + 1.3

    y0 = y0 / np.sqrt((y0**2).sum()) * radius

    # get the trajectory
    pend_ivp = scipy.integrate.solve_ivp(fun=dynamics_fn, t_span=t_span, y0=y0, t_eval=t_eval, rtol=1e-10)
    q, p = pend_ivp['y'][0], pend_ivp['y'][1]

    # add noise
    q += np.random.randn(*q.shape) * noise_std
    p += np.random.randn(*p.shape) * noise_std

    trajectory = np.vstack([q, p]).T

    return trajectory
    # return q, p

def get_dataset_slide(seed=0, samples=50, test_split=0.5, input_slide=5, output_slide=5):
    '''
    :param seed: random seed
    :param samples: sampling trajectory
    :param test_split: test data Proportionality
    :param input_slide: input slide number
    :param output_slide: output slide number
    :return: output the train and test data as a dictionary with the key "input","output","test_input","test_output"
    '''
    data = {'meta': locals()}

    # randomly sample inputs
    np.random.seed(seed)

    input, output = [], []
    for s in range(samples):
        trajectory = get_trajectory()
        n = np.shape(trajectory)[1]
        for i in range(np.shape(trajectory)[0]-input_slide-output_slide+1):
            # input = np.vstack(input, trajectory[i:i+input_slide, :])
            # output = np.vstack(output, trajectory[i+input_slide: i+input_slide+output_slide, :])
            input.append(trajectory[i:i+input_slide, :].ravel())
            output.append(trajectory[i+input_slide:i+input_slide+output_slide, :].ravel())
    data['input'] = input
    data['output'] = output

    split_ix = int(len(data['input']) * test_split)
    split_data = {}
    for k in ['input', 'output']:
        split_data[k], split_data['test_' + k] = data[k][:split_ix], data[k][split_ix:]
    data = split_data
    return data
