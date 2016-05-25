#!/usr/bin/env ipython
# -*- coding: utf-8 -*-
#
# For the Experiment class.
#
# ------------------------------------------
# author:       Stephanie Hyland (@corcra)
# date:         17/5/16
# ------------------------------------------

import numpy as np
from unitary_np import unitary_matrix
from scipy.fftpack import fft, ifft
from functools import partial

# === loss functions === #
def trivial_loss(parameters, batch):
    """
    For testing.
    Parameters is just a vector, which we add to x to get y-hat. Very simple.
    """
    x, y = batch

    y_hat = x + parameters
    differences = y_hat - y
    loss = np.mean(np.linalg.norm(y_hat - y, axis=1))
    return loss

def free_matrix_loss(parameters, batch):
    """
    For testing.
    Parameters is now a matrix!
    """
    x, y = batch
    d = x.shape[1]

    M = parameters.reshape(d, d)

    y_hat = np.dot(x, M)

    differences = y_hat - y
    loss = np.mean(np.linalg.norm(y_hat - y, axis=1))
    return loss

# === parametrisation-specific functions === #

def do_reflection(x, v_re, v_im, theano_reflection=False):
    """
    Hey, it's this function again! Woo!
    NOTE/WARNING: theano_reflection gives a DIFFERENT RESULT to the other one...
    see this unresolved issue:
    https://github.com/amarshah/complex_RNN/issues/2
    """
    if theano_reflection:
        # (mostly copypasta from theano, with T replaced by np all over)
        # FOR NOW OK
        input_re = np.real(x)
        # alpha
        input_im = np.imag(x)
        # beta
        reflect_re = v_re
        # mu
        reflect_im = v_im
        # nu

        vstarv = (reflect_re**2 + reflect_im**2).sum()

        # (the following things are roughly scalars)
        # (they actually are as long as the batch size, e.g. input[0])
        input_re_reflect_re = np.dot(input_re, reflect_re)
        # αμ
        input_re_reflect_im = np.dot(input_re, reflect_im)
        # αν
        input_im_reflect_re = np.dot(input_im, reflect_re)
        # βμ
        input_im_reflect_im = np.dot(input_im, reflect_im)
        # βν

        a = np.outer(input_re_reflect_re - input_im_reflect_im, reflect_re)
        # outer(αμ - βν, mu)
        b = np.outer(input_re_reflect_im + input_im_reflect_re, reflect_im)
        # outer(αν + βμ, nu)
        c = np.outer(input_re_reflect_re - input_im_reflect_im, reflect_im)
        # outer(αμ - βν, nu)
        d = np.outer(input_re_reflect_im + input_im_reflect_re, reflect_re)
        # outer(αν + βμ, mu)

        output_re = input_re - 2. / vstarv * (d - c)
        output_im = input_im - 2. / vstarv * (a + b)

        output = output_re + 1j*output_im
    else:
        # do it the 'old fashioned' way
        v = v_re + 1j*v_im
        # aka https://en.wikipedia.org/wiki/Reflection_%28mathematics%29#Reflection_through_a_hyperplane_in_n_dimensions
        # but with conj v dot with x
        output = x - (2.0/np.dot(v, np.conj(v))) * np.outer(np.dot(x, np.conj(v)), v)

    return output

def complex_RNN_loss(parameters, batch, permutation, theano_reflection=False):
    """
    Transform data according to the complex_RNN transformations.
    (requires importing a bunch of things and weird tensorflow hax)
    NOTE: no longer folding in any input data...

    Parameters, once again, numpy array of values.
    """
    x, y = batch
    d = x.shape[1]

    # === expand the parameters === #

    # diag1
    thetas1 = parameters[0:d]
    diag1 = np.diag(np.cos(thetas1) + 1j*np.sin(thetas1))
    # reflection 1
    reflection1_re = parameters[d:2*d]
    reflection1_im = parameters[2*d:3*d]
    # fixed permutation (get from inputs)
    # diag 2
    thetas2 = parameters[3*d:4*d]
    diag2 = np.diag(np.cos(thetas2) + 1j*np.sin(thetas2))
    # reflection 2
    reflection2_re = parameters[4*d:5*d]
    reflection2_im = parameters[5*d:6*d]
    # diag 3
    thetas3 = parameters[6*d:7*d]
    diag3 = np.diag(np.cos(thetas3) + 1j*np.sin(thetas3))

    # === do the transformation === #
    step1 = np.dot(x, diag1)
    step2 = fft(step1)
    step3 = do_reflection(step2, reflection1_re, reflection1_im, theano_reflection)
    #step3 = step2
    step4 = np.dot(step3, permutation)
    step5 = np.dot(step4, diag2)
    step6 = ifft(step5)
    step7 = do_reflection(step6, reflection2_re, reflection2_im, theano_reflection)
    #step7 = step6
    step8 = np.dot(step7, diag3)
    # POSSIBLY do relu_mod...

    # === now calculate the loss ... === #
    y_hat = step8
    differences = y_hat - y
    loss = np.mean(np.linalg.norm(y_hat - y, axis=1))
    return loss

def complex_RNN_multiloss(parameters, permutations, batch):
    """
    Transform data according to the complex_RNN transformations.
    ... but now with n*n parameters.
    We achieve this by repeating the basic components.
    
    Components:
        D   (n params)
        R   (2n params)
        F   0 params
        Pi  0 params

    Canonical form:
        U = D_3 R_2 \mathcal{F}^{-1} D_2 \Pi R_1 \mathcal{F} D_1

    ... bleh, this has no obvious pattern.

    Putting two Ds beside each other is largely uninteresting.
    """
    raise NotImplementedError
    # === split up parameters, permutations

    # === combine in components

    # === get the loss

    return loss
    
    

def general_unitary_loss(parameters, batch, basis_change=None):
    """
    Hey, it's my one! Rendered very simple by existence of helper functions. :)
    """
    x, y = batch
    d = x.shape[1]

    lambdas = parameters
    U = unitary_matrix(d, lambdas=lambdas, basis_change=basis_change)

    y_hat = np.dot(x, U.T)
    differences = y_hat - y
    loss = np.mean(np.linalg.norm(y_hat - y, axis=1))

    return loss

# === experiment class === #
class Experiment(object):
    """
    Defines an experimental setting.
    """
    def __init__(self, name, d, 
                 project=False, 
                 random_projections=0,
                 restrict_parameters=False,
                 theano_reflection=False,
                 change_of_basis=0):
        # required
        self.name = name
        self.d = d
        # with defaults
        self.project = project
        self.random_projections = random_projections
        self.restrict_parameters = restrict_parameters
        self.theano_reflection = theano_reflection
        self.change_of_basis = change_of_basis
        # check
        self.check_attributes()
        # defaults
        self.test_loss = -9999
        self.learning_rate = 0.001
        # derived
        self.set_basis_change()     # this must happen before set_loss...
        self.set_loss()
        self.set_learnable_parameters()
        # TODO (sparse)
        self.nonzero_index = None

    def check_attributes(self):
        """
        Make sure attributes are sensible.
        """
        if self.d <= 7 and self.restrict_parameters:
            print 'WARNING: d is <= 7, but restrict parameters is true. It will have no effect.'
        if self.restrict_parameters and not 'general_unitary' in self.name:
            print 'WARNING: restrict_parameters is only implemented for unitary experiments. It will have no effect.'

        if self.theano_reflection and not self.name == 'complex_RNN_vanilla':
            raise ValueError(self.theano_reflection)

        if self.name == 'projection' and not self.project:
            raise ValueError(self.project)
        
    def initial_parameters(self):
        """
        Return initial parameters for a given experimental setup.
        """
        d = self.d
        if self.name in {'trivial'}:
            ip = np.random.normal(size=d) + 1j*np.random.normal(size=d)
        elif self.name in {'free_matrix', 'projection'}:
            ip = np.random.normal(size=d*d) + 1j*np.random.normal(size=d*d)
        elif 'complex_RNN' in self.name:
            ip = np.random.normal(size=7*d)
        elif 'general_unitary' in self.name:
            ip = np.random.normal(size=d*d)
        else:
            raise ValueError(self.name)
       
        n_parameters = np.prod(ip.shape)
        assert n_parameters == self.n_parameters
        if ip.dtype == 'complex':
            n_parameters = 2*n_parameters
        print 'Initialising', n_parameters, 'real parameters.'
        return ip

    def set_loss(self):
        """
        Pick the loss function.
        """
        print '(experiment ' + self.name +'): (re)setting loss function.'
        if self.name in {'trivial'}:
            fn = trivial_loss
            self.n_parameters = self.d
        elif self.name in {'free_matrix', 'projection'}:
            fn = free_matrix_loss
            self.n_parameters = self.d*self.d
        elif 'complex_RNN' in self.name:
            permutation = np.random.permutation(np.eye(self.d))
            fn = partial(complex_RNN_loss, permutation=permutation, 
                         theano_reflection=self.theano_reflection)
            self.n_parameters = 7*self.d
        elif 'general_unitary' in self.name:
            if self.change_of_basis > 0 and self.basis_change is None:
                self.set_basis_change()
            fn = partial(general_unitary_loss, basis_change=self.basis_change)
            self.n_parameters = self.d*self.d
        else:
            raise ValueError(self.name)

        self.loss_function = fn
        return True

    def set_learnable_parameters(self):
        d = self.d
        if self.restrict_parameters:
            learnable_parameters = np.random.choice(d*d, 7*d, replace=False)
            self.learnable_parameters = learnable_parameters
        else:
            self.learnable_parameters = np.arange(self.n_parameters)
        return True

    def set_basis_change(self):
        if self.change_of_basis > 0:
            d = self.d
            scale = self.change_of_basis
            basis_change = np.random.uniform(low=-scale, high=scale, size=(d*d,d*d))
            self.basis_change = basis_change
        else:
            self.basis_change = None
        return True

# === specific experimental designs === #
def presets(d):
    """
    Returns a list of 'preset' experiment objects.
    """
    proj = Experiment('projection', d, project=True)
    complex_RNN = Experiment('complex_RNN', d)
    general = Experiment('general_unitary', d)
    exp_list = [proj, complex_RNN, general]
    if d > 7:
        general_restrict = Experiment('general_unitary_restricted', d, restrict_parameters=True)
        exp_list.append(general_restrict)
    return exp_list

def test_random_projections(d=6):
    exp_list = []
    for j in [4, 9, 16, 25, 36]:
        #    for j in np.linspace(np.sqrt(d), 0.5*d*(d-1), num=3, dtype=int):
        exp_list.append(Experiment('general_unitary_' + str(j), d, random_projections=j))
    exp_list.append(Experiment('general_unitary', d))
    return exp_list

def basis_change(d):
    """ testing how the change of basis influences learning """
    general_default = Experiment('general_unitary', d)
    general_basis_1 = Experiment('general_unitary_basis10', d, change_of_basis=10)
    general_basis_2 = Experiment('general_unitary_basis50', d, change_of_basis=50)
    exp_list = [general_default, general_basis_1, general_basis_2]
    return exp_list

def rerun(d):
    """
    Steph is silly.
    """
    proj = Experiment('projection', d, project=True)
    complex_RNN = Experiment('complex_RNN', d)
    general = Experiment('general_unitary', d)
    general_basis_5 = Experiment('general_unitary_basis5', d, change_of_basis=5)
    exp_list = [proj, complex_RNN, general, general_basis_5]
    if d > 7:
        general_restricted = Experiment('general_unitary_restricted', d, restrict_parameters=True)
        exp_list.append(general_restricted)
    return exp_list
