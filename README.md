# Introduction 

This package modifies the popular [deep_q_rl](https://github.com/spragunr/deep_q_rl) implementation of the deep
Q-learning algorithm described in:

[Playing Atari with Deep Reinforcement Learning](http://arxiv.org/abs/1312.5602)
Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis
Antonoglou, Daan Wierstra, Martin Riedmiller

and 

Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning." Nature 518.7540 (2015): 529-533.

to play the game of Go. Our main work involved setting up [GNU Go](https://www.gnu.org/software/gnugo/) as an opponent, and implementing techniques from:

Clark, Christopher. Storkey, Amos. [Training Deep Convolutional Neural Networks to Play Go](http://jmlr.org/proceedings/papers/v37/clark15.html), ICML 2015

For more details, see our [report](https://github.com/ssampang/deep_q_rl/blob/master/Playing%20Go%20with%20Deep%20Learning.pdf).

# Dependencies

* A reasonably modern NVIDIA GPU
* OpenCV
* [Theano](http://deeplearning.net/software/theano/) ([https://github.com/Theano/Theano](https://github.com/Theano/Theano))
* [Lasagne](http://lasagne.readthedocs.org/en/latest/) ([https://github.com/Lasagne/Lasagne](https://github.com/Lasagne/Lasagne)
* [Pylearn2](http://deeplearning.net/software/pylearn2/) ([https://github.com/lisa-lab/pylearn2](https://github.com/lisa-lab/pylearn2))

The script `dep_script.sh` can be used to install all dependencies under Ubuntu.

