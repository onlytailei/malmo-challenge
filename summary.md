# **bacon-reloaded**

This is **bacon-reloaded**! Here's our summary for our work for **The Malmo Collaborative AI Challenge**

## Approach
* Asynchronous Advantage Actor-Critic (A3C) with Generalized Advantage Estimation (GAE)
> We implemented A3C [[1]](https://arxiv.org/abs/1602.01783) along with GAE [[2]](https://arxiv.org/abs/1506.02438) to train our agent. We choose to use A3C mainly due to the observation that the data is relatively expensive to collect from the malmo environment, thus we want to be able to collect experiences over multiple machines.

## Novel Points
* Training over multiple machines with docker
> pass
* Curriculum Learning
> After training a vanilla A3C agent on the malmo challenge, we found that the agent is tended to go to the exit directly and get the small reward instead of trying to find a way to coorperate with its opponent and catch the pig. We suspect that this is due to the agent rarely sees such experiences as in the latter case so it cannot learn sufficiently from such good experiences. We then modify our training procedure so that in the beginning of training, the opponent would always be executing A* actions, so the probability that our agent and the opponent could catch the pig together would be higher. The probability that the opponent is a ``RandomAgent`` instead of an ``AStarAgent`` is linearly annealed from ``0`` to ``0.75`` (which is the original setting in the malmo challenge).
* Periodical Finetuning
> One thing we notice is that the malmo environment is relatively not stable enough when running on remote clusters: we start 8 processes (2 malmo environment each) then many of them stop to communicate ... in about 10 minutes. Thus our solution is to restart a new training instance every 10 minutes, by finetuning from the model saved from the last training instance.
