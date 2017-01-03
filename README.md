Think up a better name for me. OpenAI Universe wouldbe RL trainer

### What is this thing?

__The project is in an early stage of development__

This is going to be a deep reinforcement learning platform for OpenAI universe or whatever you can formulate as an MDP.

The idea is to make dedicated processes that constantly interact with the environment and record sessions under current policy, while other dedicated processes that sample those sessions and update the policy.

So, the key components are
* __player process__ that takes policy params (NN weights) interacts and records sessions. Several such processess run in parallel.
* __learner process__ that updates policy params (you guesses it, NN weights) based on the recorded sessions
* __database__ server that stores sessions and NN weights


![img](https://s23.postimg.org/cei1cd2iz/tinyverse_scheme.png)


### Long version

__An embarassingly parallel Universe and Gym solver__

Not so far from now, OpenAI released the Universe platform, which basically wraps the world into an MDP interface. Apart from that, many other environments can be easily wrapped this way, opening possibilities to automate pretty much anything human with a PC is capable of.

The only problem is, one no longer can use the standard RL pipelines for such task, since
- the environment can't be 'paused' (especially the MMO games), you can't stop mid-game to train the network;
- simulation typically takes time, would be great to run several simulrations in parallel (esp. on multiple machines)
- many environments are partially observable, so one can't just store a collection of (s,a,r,s,[a]) tuples,

So, here we are, trying to pull together something that trains in such weird conditions. If we happen to succeed, the result may also be useful to speed up training for conventional RL problems like atari, especially if one has loads of [weak] computers.


