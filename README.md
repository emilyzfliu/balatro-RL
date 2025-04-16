# TODO
- Refactor balatro env
- Expand `balatro_game.py` to include all eight rounds of the standard game.
- Modify `BalatroGame` implementation to account for Jokers and special cards.
- Simulate shop task.


# Directory overview

## Algorithms
**Greedy baselines:** Simulates playing a single round of Balatro using both a greedy and an epsilon-greedy approach to select the most optimal hand.

# Approach

## Two-phase approach
We formulate Balatro as a cooperative game between two agents: `round`, which plays cards for each round, and `shop`, which makes purchases in the shop at the end of each round to improve the card deck.

| Agent | Approach | Inputs | Modifications to environment |
|-------|----------|---------|----------------------------|
| `round` | Epsilon-greedy, Policy gradient | Deck, round/ante, Jokers, Tarot cards | Has the opportunity to upgrade playing cards through the use of Tarot cards. |
| `shop` | Random search, Policy gradient | Current money, shop options: Two cards, two booster packs, one voucher. Unlimited reroll opportunities. | Changes current money, adds/removes cards, applies jokers and modifiers |


# Balatro Gym
This project was originally forked from https://github.com/cassiusfive/balatro-gym.

Descrption:

> `balatro-gym` provides a [Gymnasium](https://gymnasium.farama.org/) environment for the poker-themed rougelike deck-builder [Balatro](https://www.playbalatro.com/). This project provides a standard interface to train reinforcement learning models for Balatro v1.0.0.
