from abc import ABC, abstractmethod
from enum import Enum
import random
import numpy as np
from typing import List, Tuple
from collections import Counter
import json


WORD_LENGTH = 5
VERBOSE = 1
with open("./solutions.txt", "r") as f:
    solutions = f.readlines()
with open("./words.txt", "r") as f:
    guesses = f.readlines()


def clean_words(lines):
    """Cleans input of wordle words"""
    return [i.strip() for i in lines[0].replace('"', "").split(",")]


SOLUTIONS = clean_words(solutions)
GUESSES = clean_words(guesses)


class CharacterResult(Enum):
    NOT_EVALUATED = 0
    INCORRECT = 1
    WRONG_POSITION = 2
    CORRECT = 3


class ResponseCache:
    """A caching class which saves responses against guesses when benchmarking"""

    cache = {}
    cur_cache_level = cache

    def __init__(self) -> None:
        pass

    def add(self, response, guess):
        self.cur_cache_level[str(response)] = {}
        self.cur_cache_level[str(response)]["guess"] = guess

    def get(self, response):
        return self.cur_cache_level[str(response)]["guess"]

    def get_cur_keys(self):
        return self.cur_cache_level.keys()

    def change_level(self, response):
        self.cur_cache_level = self.cur_cache_level[str(response)]

    def go_to_top(self):
        self.cur_cache_level = self.cache


class WordleGame:
    """Methods for playing a game of Wordle"""

    def __init__(self) -> None:
        self.guesses = 0
        self.solution = ""

    def reset(self):
        self.guesses = 0

    def set_solution(self, solution: str) -> None:
        self.solution = solution

    @classmethod
    def get_response(cls, guess: str, solution: str) -> List[CharacterResult]:
        """Get the response to a guess, given a solution"""

        response = [CharacterResult.NOT_EVALUATED] * WORD_LENGTH
        char_apparences = {char: solution.count(char) for char in solution}
        idx_to_reevaluate = []

        # Evaluate all the correct guesses first, as the number of appearences of a character matters for Wordle evaluation
        # i.e. If the word was AAQQQ, and I guessed QQAAA, Wordle would register this as [wrong_loc, wrong_loc, wrong_loc, wrong_loc, incorrect]
        # Note how the last character is specifically incorrect, as there are only 2 A's in the correct word.
        for index, char in enumerate(guess):
            if solution[index] == char:
                response[index] = CharacterResult.CORRECT
                char_apparences[char] -= 1
            else:
                idx_to_reevaluate.append(index)

        # Evaluate the incorrect and wrong position guesses
        for index in idx_to_reevaluate:
            char = guess[index]
            if (char in solution) and (char_apparences[char] > 0):
                response[index] = CharacterResult.WRONG_POSITION
                char_apparences[char] -= 1

            else:
                response[index] = CharacterResult.INCORRECT

        return response

    @classmethod
    def calculate_possible_responses(cls, guess: str, solutions: Tuple[str]) -> Tuple[List[Tuple[CharacterResult]], List[float]]:
        """Helper method to calculate all possible responses to a guess, given a list of possible solutions"""
        responses = []

        for solution in solutions:
            responses.append(cls.get_response(guess, solution))

        # Get the unique responses, as well as the number of different ways those responses can appear.
        responses_tuple = map(tuple, responses)
        counts = Counter(responses_tuple)
        responses = list(counts.keys())
        total_counts = sum(counts.values())
        probs = [i / total_counts for i in counts.values()]
        return responses, probs


class AbstractWordleStrat(ABC):
    """An abstract base class to create Wordle strategies from. Contains helper functions"""

    def __init__(self, possible_guesses: list, possible_solutions: list) -> None:
        self.possible_guesses = (
            possible_guesses  # Keep track of possible guesses we're allowed to make
        )
        self.possible_solutions = possible_solutions  # Keep track of the possible solutions to the current problem.

    def _less_than_max_chars(self, max_appearences, possible_solution):
        for char, apparences in max_appearences.items():
            if possible_solution.count(char) != apparences:
                return False
        return True

    def _more_than_min_chars(self, min_appearences, possible_solution):
        for char, apparences in min_appearences.items():
            if possible_solution.count(char) < apparences:
                return False
        return True

    def _responses_match_word(self, guess, response, possible_solution):
        for index, char_response in enumerate(response):
            if char_response == CharacterResult.WRONG_POSITION:
                if (guess[index] not in possible_solution) or (
                    possible_solution[index] == guess[index]
                ):
                    return False

            if char_response == CharacterResult.CORRECT:
                if possible_solution[index] != guess[index]:
                    return False

        return True

    def reduce_possible_solutions(self, guess: str, response: List[CharacterResult]):
        """Given a guess and response to it, return the new list of possible solutions."""

        # Remove solutions with known answers
        new_solutions = []

        # Get the minimal apparences for each character
        min_appearences = {char: 0 for char in guess}
        max_appearences = {}
        idx_to_reevaluate = []
        for index, char_response in enumerate(response):
            if char_response != CharacterResult.CORRECT:
                idx_to_reevaluate.append(index)
                continue
            char = guess[index]
            min_appearences[char] += 1

        # Calculate the minimum and maximum appearences of characters, if possible
        for index in idx_to_reevaluate:
            char_response = response[index]
            char = guess[index]
            if char_response == CharacterResult.INCORRECT:
                max_appearences[char] = min_appearences[char]
            else:
                min_appearences[char] += 1

        for possible_solution in self.possible_solutions:

            # Run tests to see if each word passes the conditions imposed by the response
            if not self._responses_match_word(guess, response, possible_solution):
                continue
            if not self._less_than_max_chars(max_appearences, possible_solution):
                continue
            if not self._more_than_min_chars(min_appearences, possible_solution):
                continue

            # If everything passes, add it to the possible word list
            new_solutions.append(possible_solution)

        return new_solutions

    @abstractmethod
    def get_next_guess(self) -> str:
        pass

    @abstractmethod
    def reset(self):
        pass


class StratMaxEntropy(AbstractWordleStrat):
    """Guesses each next word by trying to maximum the expected entropy of each guess"""

    def __init__(self, possible_guesses: list, possible_solutions: list) -> None:
        super().__init__(possible_guesses, possible_solutions)

        self.solutions_copy = (
            self.possible_solutions.copy()
        )  # Used for benchmarking to keep track of solutions

    def reset(self):
        self.possible_solutions = self.solutions_copy

    def get_next_guess(self):
        """Used to get the next guess, given the list of possible guesses and possible solutions"""

        n_current_solutions = len(self.possible_solutions) * 1.0

        # If we've only got two possible solutions, we may as well guess the first one rather than a random guess
        if n_current_solutions <= 2:
            return self.possible_solutions[0]

        guess_entropy = (
            {}
        )  # Collect the entropies of every guess + solution combination
        guess_minmax = {}  # Collect the entropies of every guess + solution combination
        guesses = self.possible_solutions + self.possible_guesses

        # For all possible guesses, and for all possible responses to that guess, calculate the entropy
        for n, guess in enumerate(guesses):

            if VERBOSE:
                print(f"- - - {n}/{len(guesses)} - - - ", end="\r")

            guess_entropy[guess] = 0
            guess_minmax[guess] = []

            responses, probs = WordleGame.calculate_possible_responses(
                guess, self.possible_solutions
            )

            for idx, response in enumerate(responses):
                new_solutions = self.reduce_possible_solutions(guess, response)
                im = np.log2(n_current_solutions / len(new_solutions))
                guess_entropy[guess] += im * probs[idx]
                guess_minmax[guess].append(im * probs[idx])

        if VERBOSE > 1:
            for k, v in sorted(guess_minmax.items(), key=lambda x: sum(x[1])):
                print(k, v)

        guess_vals = [(k, sum(v)) for k, v in guess_minmax.items()]
        listy = np.array(guess_vals)
        words = listy[:, 0]
        values = listy[:, 1].astype(float)
        winner_idx = np.argwhere(values == np.amax(values))
        if len(winner_idx) > 1:
            tie_break = []
            for k in words[winner_idx]:
                word = k[0]
                tie_break.append((word, min(guess_minmax[word])))
            next_guess = max(tie_break, key=lambda x: x[1])[0]
        else:
            next_guess = max(guess_vals, key=lambda x: x[1])[0]

        return next_guess


class StratMinMax(AbstractWordleStrat):
    """Guesses each next word by selecting the best, worst case scenario"""

    def __init__(self, possible_guesses: list, possible_solutions: list) -> None:
        super().__init__(possible_guesses, possible_solutions)

        self.solutions_copy = (
            self.possible_solutions.copy()
        )  # Used for benchmarking to keep track of solutions

    def reset(self):
        self.possible_solutions = self.solutions_copy

    def get_next_guess(self):
        """Used to get the next guess, given the list of possible guesses and possible solutions"""

        n_current_solutions = len(self.possible_solutions) * 1.0

        # If we've only got two possible solutions, we may as well guess the first one rather than a random guess
        if n_current_solutions <= 2:
            return self.possible_solutions[0]

        guess_minmax = {}  # Collect the entropies of every guess + solution combination
        guesses = self.possible_solutions + self.possible_guesses

        # For all possible guesses, and for all possible responses to that guess, calculate the entropy
        for n, guess in enumerate(guesses):

            if VERBOSE:
                print(f"- - - {n}/{len(guesses)} - - - ", end="\r")

            guess_minmax[guess] = []

            responses, probs = WordleGame.calculate_possible_responses(
                guess, self.possible_solutions
            )

            for idx, response in enumerate(responses):
                new_solutions = self.reduce_possible_solutions(guess, response)
                im = np.log2(n_current_solutions / len(new_solutions))
                guess_minmax[guess].append(im)

        if VERBOSE > 1:
            for k, v in sorted(guess_minmax.items(), key=lambda x: sum(x[1])):
                print(k, v)

        guess_vals = [(k, min(v)) for k, v in guess_minmax.items()]
        next_guess = max(guess_vals, key=lambda x: x[1])[0]

        return next_guess


class WordleBot:
    """A class which, when provided a strategy, can play wordle"""

    class BotMethods(Enum):
        interactive = 0
        benchmark = 1
        manual = 2
        automatic = 3

    def __init__(self, strategy: AbstractWordleStrat, method: BotMethods = BotMethods.interactive) -> None:
        self.strategy = strategy
        self.method = method  # Can be interactive, benchmarking, or manual
        self.game = WordleGame()

        self.guesses = 0
        self.cache = ResponseCache()  # A cache to store results when benchmarking

    def start(self):
        """Starts the WordleBot and plays games"""
        if self.method == self.BotMethods.benchmark:
            self._benchmark()
        elif self.method == self.BotMethods.manual:
            print("=== Please enter the solution ===")
            self.game.set_solution(input().lower())
            self._play_single_game()
        elif self.method == self.BotMethods.interactive:
            self._play_single_game()

    def _reset(self):
        """Used for resetting the game and strategy in between games"""
        self.strategy.reset()
        self.game.reset()

    def _benchmark(self):
        """Run the WordleBot on all possible solutions, and calculate and store the results"""
        self.results = {}
        solutions_copy = self.strategy.possible_solutions.copy()
        random.shuffle(solutions_copy)

        # Run through all possible solutions
        for n, solution in enumerate(solutions_copy):
            self._reset()

            self.game.set_solution(solution)
            guesses = self._play_single_game()
            self.results[solution] = guesses
            print(f"{solution.upper()} took {guesses} guesses")
            print(f"Cur Average = {np.mean(list(self.results.values()))}.  {n}/{len(solutions_copy)}")
            print(" ")

            # Store the results
            with open("results.json", "w") as f:
                json.dump(self.results, f)
        return self.results

    def _play_single_game(self) -> int:
        """Play a single game of Wordle"""
        response = [CharacterResult.NOT_EVALUATED] * WORD_LENGTH

        while True:

            if self.method == self.BotMethods.manual:
                guess = input()
                if guess == "":
                    guess = self.strategy.get_next_guess()
            else:
            # Check our cache to see if we've seen this response, given the previous guesses and responses
                if str(response) in self.cache.get_cur_keys():
                    # If yes, get the next guess from the cache
                    guess = self.cache.get(response)
                else:
                    # If no, calculate the next guess and store it in the cache
                    guess = self.strategy.get_next_guess()
                    self.cache.add(response, guess)
                self.cache.change_level(response)  # Drop into the next level of the cache

            if VERBOSE:
                print(f"=== Best next guess is {guess.upper()} ===")

            self.game.guesses += 1

            if self.method == self.BotMethods.interactive:
                # If playing interactively, allow the user to enter the response
                mapping = {
                    "0": CharacterResult.INCORRECT,
                    "1": CharacterResult.WRONG_POSITION,
                    "2": CharacterResult.CORRECT,
                }

                print(
                    "=== Enter response in correct order (0 = incorrect, 1=wrong position, 2=correct) ===  "
                )
                for idx, i in enumerate(input()):
                    response[idx] = mapping[i]
            else:
                print(guess)
                response = WordleGame.get_response(guess, self.game.solution)

            # If we've won, exit
            if response == [CharacterResult.CORRECT] * WORD_LENGTH:
                break

            # Otherwise, reduce the possible solutions list
            self.strategy.possible_solutions = self.strategy.reduce_possible_solutions(
                guess, response
            )
            if VERBOSE:
                print(self.strategy.possible_solutions)

        if VERBOSE:
            print(f"=== Game took {self.game.guesses} guesses ===  ")

        # Go back to the top of the cache
        self.cache.go_to_top()

        return self.game.guesses


def main():

    max_entropy_strat = StratMaxEntropy(
        possible_solutions=SOLUTIONS[:], possible_guesses=GUESSES[:]
    )
    minmax_strat = StratMinMax(possible_solutions=SOLUTIONS[:], possible_guesses=GUESSES[:])
    wordle_bot = WordleBot(strategy=max_entropy_strat, method=WordleBot.BotMethods.interactive)

    # Add the first word to the cache
    wordle_bot.cache.add([CharacterResult.NOT_EVALUATED]*5, 'reast')
    wordle_bot.start()


if __name__ == "__main__":
    main()
