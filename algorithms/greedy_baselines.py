import sys
sys.path.append('.')
from balatro_gym.balatro_game import BalatroGame, Card
from utils import *
import random

# Because discarding adds zero reward, the greedy approach is to play the best hand possible
def greedy(game: BalatroGame, verbose: bool = True) -> list[Card]:
    # A pure greedy approach has a 5% win rate in the first round.
    while game.state == BalatroGame.State.IN_PROGRESS:
        if verbose:
            print('Current hand:', game.hand_to_string())
        all_hand_indices = enumerate_possible_hand_indices(game.hand_size)
        best_score = 0
        best_hand = []
        for hand_indices in all_hand_indices:
            try:
                cards = []
                for card_index in hand_indices:
                    if card_index >= len(game.hand_indexes):
                        raise IndexError(f"Card index {card_index} out of range for hand_indexes of length {len(game.hand_indexes)}")
                    cards.append(game.deck[game.hand_indexes[card_index]])
                score = game._evaluate_hand(cards)
                if score > best_score:
                    best_score = score
                    best_hand = hand_indices
            except IndexError as e:
                if verbose:
                    print(f"Skipping invalid hand combination: {e}")
                continue
            except Exception as e:
                print(f"Unexpected error: {e}")
                print(hand_indices)
                print(game.hand_indexes)
                raise e

        for card_index in best_hand:
            game.highlight_card(card_index)
        
        if verbose:
            print('Best hand:', game.highlighted_to_string())
        
        score = game.play_hand()
        if verbose:
            print('Score:', score)
            print('Round score:', game.round_score)
    # print(game.state)
    return game.state

def epsilon_greedy(game: BalatroGame, epsilon: float = 0.1, verbose: bool = True) -> list[Card]:
    while game.state == BalatroGame.State.IN_PROGRESS:
        if verbose:
            print('Current hand:', game.hand_to_string())
        all_hand_indices = enumerate_possible_hand_indices(game.hand_size)
        
        best_score = 0
        best_hand = []
        for hand_indices in all_hand_indices:
            try:
                cards = []
                for card_index in hand_indices:
                    if card_index >= len(game.hand_indexes):
                        raise IndexError(f"Card index {card_index} out of range for hand_indexes of length {len(game.hand_indexes)}")
                    cards.append(game.deck[game.hand_indexes[card_index]])
                score = game._evaluate_hand(cards)
                if score > best_score:
                    best_score = score
                    best_hand = hand_indices
            except IndexError as e:
                if verbose:
                    print(f"Skipping invalid hand combination: {e}")
                continue
            except Exception as e:
                print(f"Unexpected error: {e}")
                print(hand_indices)
                print(game.hand_indexes)
                raise e
        
        if random.random() < epsilon:
            hand_indices = random.choice(all_hand_indices)
            discard = True if random.random() < 0.5 else False
        else:
            hand_indices = best_hand
            discard = False

        for card_index in hand_indices:
            game.highlight_card(card_index)
        
        if verbose:
            print('Selected hand:', game.highlighted_to_string())
        
        if discard:
            game.discard_hand()
            if verbose:
                print('Discarded hand')
        else:
            score = game.play_hand()
            if verbose:
                print('Score:', score)
                print('Round score:', game.round_score)
    # print(game.state)
    return game.state

if __name__ == "__main__":
    n_trials = 1000
    wins = 0
    losses = 0
    for _ in range(n_trials):
        game = BalatroGame()
        greedy(game, verbose=(n_trials == 1))
        if game.state == BalatroGame.State.WIN:
            wins += 1
        else:
            losses += 1
    print('Greedy win rate:', wins / n_trials)
    print('Wins:', wins, 'Losses:', losses)

    wins = 0
    losses = 0
    for _ in range(n_trials):
        game = BalatroGame()
        epsilon_greedy(game, epsilon=0.1, verbose=(n_trials == 1))
        if game.state == BalatroGame.State.WIN:
            wins += 1
        else:
            losses += 1
    print('Top k epsilon greedy win rate:', wins / n_trials)
    print('Wins:', wins, 'Losses:', losses)