# TODO
- Simulate shop task: Modify `two_phase_env.py`. Shop actions may not necessarily take one round to complete. Include a 'terminate shop' action, upon which the round is played and a positive reward earned.
- Add: All Jokers, Tarot Cards, Planet Cards to actions. Calculate full action space.
- Modify `BalatroGame` implementation to account for Jokers and special cards.
- Expand `balatro_game.py` to include all eight rounds of the standard game.


# Directory overview

## Algorithms
**Greedy baselines:** Simulates playing a single round of Balatro using both a greedy and an epsilon-greedy approach to select the most optimal hand.

# Approaches

## Two-phase approach
We formulate Balatro as a cooperative game between two agents: `round`, which plays cards for each round, and `shop`, which makes purchases in the shop at the end of each round to improve the card deck.

| Agent | Approach | Inputs | Modifications to environment |
|-------|----------|---------|----------------------------|
| `round` | Epsilon-greedy, Policy gradient | Deck, round/ante, Jokers, Tarot cards | Has the opportunity to upgrade playing cards through the use of Tarot cards. |
| `shop` | Random search, Policy gradient | Current money, shop options: Two cards, two booster packs, one voucher. Unlimited reroll opportunities. | Changes current money, adds/removes cards, applies jokers and modifiers |

### `round`
In this approach, we make the assumption that an epsilon-greedy approach, where the optimal hand is played with probability $1 - \epsilon$, will be sufficient to pass any given round. Although this assumption is not universally true (it may be more optimal to, for example, discard cards in the event that the optimal hand does not yield a high enough payoff), it is reasonable in that the majority of sustainable gameplay in later rounds is obtained through deck-building, rather than strategic gameplay within the actual round itself.

Each round's reward is defined by the monetary payout of the round in the case of a win, with a loss being a large negative reward.

### `shop`



## LLM-based approach
An alternative to the two-phase approach involves using a large language model (LLM) to play all phases of the game.

The input to the language model must contain the same inputs as the two-phase approach, represented in natural language. The language model will then output the next step to the game.

Since we are able to provide the LLM with context regarding the overarching game, as well as descriptions of various Jokers and cards, we expect a pretrained LLM to achieve decent baseline performance on this task.

Pros of LLM-based approach:
- Leverages pre-existing knowledge about poker hands, card game strategies, and general game mechanics
- Can understand and utilize complex card interactions and special abilities through natural language descriptions
- More flexible in handling rule changes or new game mechanics without retraining
- Can potentially explain its decision-making process in human-readable format
- May discover creative strategies by drawing on broader gaming knowledge

Cons of LLM-based approach:
- Higher computational cost per inference compared to trained RL models
- Difficult to guarantee optimal play without extensive prompt engineering
- Cannot easily learn from experience or improve through self-play
- May struggle with precise numerical calculations needed for optimal strategy
- Requires careful prompt engineering to maintain consistent decision-making

# Balatro Gym
This project was originally forked from https://github.com/cassiusfive/balatro-gym.

Descrption:

> `balatro-gym` provides a [Gymnasium](https://gymnasium.farama.org/) environment for the poker-themed rougelike deck-builder [Balatro](https://www.playbalatro.com/). This project provides a standard interface to train reinforcement learning models for Balatro v1.0.0.
