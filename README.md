## Reinforcement-Learning

# Terminologies Involved in RL:

Reward: Scalar Feedback Signal to indicate how well the agent is doing.

Reward Hypothesis for RL: All goals can be described by the maximisation of expected
cumulative reward

Agent:

- Executes Action

- Recieves Observation

- Recieves Reward

Environment:

- Recieves Action

- Emits Observation

- Emits Reward

History: Seqeunce of Observations, Actions, Rewards

State is a function of history and has information used to determine what happens next: $S_t = f(H_t)$

Markov State: Contains all useful information from history

State is markov if $P[S_{t+1} | S_{t}] = P[S_{t+1} | S_1, S_2 , \ldots , S_t]$

Policy is agent's behavior, i.e., map from state to action.

Value Function: Prediction of future reward for a given state.

## Markov Processes

For a Markov state s and successor state s′, the state transition
probability is defined by:

$$P_{ss} = P[S_{t+1} = s | S_t = s]$$

## Markov Processes

A Markov process is a memoryless random process, i.e. a sequence
of random states S1, S2, ... with the Markov property.

It is defined by a tuple $(S, P)$ where $S$ is a finite set of states and $P$ is the transition probability matrix.

## Markov Reward Processes

Markov Chain with values.

A tuple $(S, P, R, \gamma )$ where,

$R$ is a reward function, $R_ss' = E[R_{t+1} | S_t = s]$

$\gamma$ is discount factor, a value between 0 and 1.

Return: The return $G_t$ is the total discounted reward from time-step t.

$$G_t = \sum_{i=t+1}^{\infty} \gamma^{i-t-1}R_{i}$$

Value Function: Gives the long term reward of state $s$.

$$v(s) = E[G_t | S_t = s]$$

**Bellman Equation**

$$v(s) = E[R_{t+1} + \gamma v(S_{t+1}) | S_t = s]$$

$$v(s) = R_s + \gamma \sum_{s' \in S} P_{ss'}v(s')$$

Which can be written in matrix form as

$$v = R + \gamma P v$$

And the solution is:

$$v = (I - \gamma P)^{-1}R$$


## Markov Decision Process

A Markov decision process (MDP) is a Markov reward process with decisions

Defined as a tuple $(S,A,P,R, \gamma)$

Where,

$A$ is a finite set of actions

$P_{ss'}^a = P[S_{t+1} = s' | S_t = s, A_t=a]$

Policy: $\pi$ is a distribution over actions given states

A policy fully defines the behaviour of an agent

**Bellman Expectation Equation**

For state value function:

$$v_{\pi}(s) = E_{\pi}[R_{t+1} + \gamma v_{\pi}(S_{t+1}) | S_t = s]$$

For action value function:

$$q_{\pi}(s,a) = E_{\pi}[R_{t+1} + \gamma q_{\pi}(S_{t+1}, A_{t+1}) | S_t = s, A_t = a]$$

In matrix form:

$$v_{\pi} = R^{\pi} + \gamma P^{\pi}v_{\pi}$$

With solution

$$v_{\pi} = (I - \gamma P^{\pi})^{-1}R^{\pi}$$

**Bellman Optimality Equation** 

$$v_{*}(s) = \text{max}_a R_{s}^{a} + \gamma \sum _{s' \in S} P_{ss'}^{a}v_{*}(s')$$

## Dynamic Programming

Method of breaking complex problems into subproblems, solving those and combining the solutions.

Used in MDP to either predict value function given policy or find optimal value function and optimal policy for given MDP

For predicting value function: Iteratively apply Bellman Expectation equation on all states for the given policy

$$v^{k+1} = R^{\pi} + \gamma P^{\pi} v^{k}$$

To improve the policy (Policy Iteration): 

- Evaluate the policy

- Improve the policy by acting greedily with respect to $v_{\pi}$

- Perform this iteration many times so that policy converges to $\pi^{*}$

- Once improvement stops, Bellman optimality equation has been satisfied

Principle of optimality: 

A policy $\pi(a|s)$ achieves the optimal value from state $s$, $v_{\pi}(s) = v_{*}(s)$ if and only if

- $\pi$ achieves the optimal value from state $s'$, $v_{\pi}(s')=v_{*}(s')$

## Monte Carlo Reinforcement Learning

Model free learning, no knowledge of MDP transitions/rewards

In Monte Carlo, value = mean return

Goal is to learn $v_{\pi}$ given a policy $\pi$

Monte-Carlo policy evaluation uses empirical mean return instead of expected return

For each iteration:

- Increament counter by one: $N(s) \larr N(s) + 1$

- Increment total return $S(s) \larr S(s) + G_{t}$

- Value is mean return $V(s) \larr S(s) / N(s)$

## Temporal Difference Learning

Learns directly from incomplete episodes by updating a guess.

$$V(S_t) \larr V(S_t) + \alpha(G_t - V(S_t))$$

For TD(0):

$$G_{t} = R_{t+1} + \gamma V(S_{t+1})$$