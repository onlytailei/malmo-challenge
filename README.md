# The Malmo Collaborative AI Challenge

# **bacon-reloaded**

This is **bacon-reloaded**:P Here's our summary for our work for [**The Malmo Collaborative AI Challenge**](https://www.microsoft.com/en-us/research/academic-program/collaborative-ai-challenge/#).

## Approach
* Asynchronous Advantage Actor-Critic (A3C) with Generalized Advantage Estimation (GAE)
> We implemented A3C [[1]](https://arxiv.org/abs/1602.01783) along with GAE [[2]](https://arxiv.org/abs/1506.02438) to train our agent. We choose to use A3C mainly due to the observation that the data is relatively expensive to collect from the malmo environment, thus we want to be able to collect experiences over multiple machines.

## Novel Points
* Training over multiple machines with docker
> pass
* Curriculum Learning
> After training a vanilla A3C agent on the malmo challenge, we found that the agent is tended to go to the exit directly and get the small reward instead of trying to find a way to coorperate with its opponent and catch the pig. We suspect that this is due to the probability for the latter scenario to happen is relatively small, so the agent would have too few such examples to learn sufficiently how to get the big reward for catching the pig. This observation inspired us to utilize currilum learning in the training of our agent, in the belief that it would be much more probable for the agent to learn the optimal behavior if it starts from relatively simple tasks (by simple we mean that the probability for the agent to catch the pig is higher) then gradually transits to more difficult tasks [[3]](http://dl.acm.org/citation.cfm?id=1553380). We thus modify our training procedure so that in the beginning of training, the opponent would always be executing A* actions, so the probability that our agent and the opponent could catch the pig together would be higher. The probability that the opponent is a ``RandomAgent`` instead of an ``AStarAgent`` is linearly annealed from ``0`` to ``0.75`` (which is the original setting in the malmo challenge). This gives up a big performance gain even though we have only trained this paradigm for (one day???) iterations; we believe this training procedure can result in much better performance if we have trained it for more iterations.
* Periodical Finetuning
> One thing we notice is that the malmo environment is relatively not stable enough when running on remote clusters: we start 8 processes (2 malmo environment each) then many of them stop to communicate ... in about 10 minutes. Thus our solution is to restart a new training instance every 10 minutes, by finetuning from the model saved from the last training instance.

> A3C multiprocess in Docker

## Evaluation Results

## Video

## How to reproduce the result

## References:
* [1] [Asynchronous Methods for Deep Reinforcement Learning] (https://arxiv.org/abs/1602.01783)
* [2] [High-Dimensional Continuous Control Using Generalized Advantage Estimation] (https://arxiv.org/abs/1506.02438)
* [3] [Curriculum learning] (http://dl.acm.org/citation.cfm?id=1553380)
