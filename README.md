## Reinforcement-Learning

# Terminologies Involved in RL:

Reward: Scalar Feedback Signal to indicate how well the agent is doing.

Reward Hypothesis for RL: All goals can be described by the maximisation of expected
cumulative reward

Agent:

Executes Action

Recieves Observation

Recieves Reward

Environment:

Recieves Action

Emits Observation

Emits Reward

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
