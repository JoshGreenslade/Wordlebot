import itertools
import numpy as np
import time
import pprint
from collections import Counter


= ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']


def get_response(p_i_1, gki, solution):
    response = [0]*len(gki)
    keys, vals = np.unique(solution, return_counts=True)
    appearences = dict(zip(keys, vals))
    for index, char in enumerate(gki):
        if solution[index] == char:
            response[index] = 1
            appearences[char] -= 1
    for index, char in enumerate(gki):
        if (char in solution) and (appearences[char] > 0) and (response[index] != 1):
            response[index] = 2
            appearences[char] -= 1
        elif (char not in solution): 
            response[index] = 0
    return response

def reduce_pool(p_i_1, gki, r_j_i):
    
    p_i_1 = p_i_1.copy()
        
    # Remove solutions with known answers
    to_remove = set()

    # Get the minimal appearasnces for each character
    min_appearences = {}
    max_appearences = {}
    for c in set(gki):
        response_indexes = np.where([i == c for i in gki])
        for i in response_indexes:
            m_app = sum([r_j_i[j] > 0 for j in i])
            if m_app > 0:
                min_appearences[c] = m_app
            if (0 in [r_j_i[j] for j in i]):
                if (1 in [r_j_i[j] for j in i]) or (2 in [r_j_i[j] for j in i]):
                    max_appearences[c] = min_appearences[c]
    keys_to_check = list(set(list(min_appearences.keys()) + list(max_appearences.keys())))
    for possible_solution in p_i_1:
        
        char_apps = {i:possible_solution.count(i) for i in keys_to_check}
        unique_chars = set(possible_solution)
        for char, amount in min_appearences.items():
            if char_apps[char] < amount:
                to_remove.add(possible_solution)
                continue
        for char, amount in max_appearences.items():
            if char_apps[char] > amount:
                to_remove.add(possible_solution)
                continue

        for index, r in enumerate(r_j_i):
            char = gki[index]
            # Remove solution where answer is known
            if (r == 1) and (possible_solution[index] != char):
                to_remove.add(possible_solution)
                break
                
            elif (r == 0) and (char in possible_solution):
                if char not in min_appearences.keys():
                    to_remove.add(possible_solution)
                    break
                if min_appearences[char] == 0:
                    to_remove.add(possible_solution)
                    break
                
            elif (r == 2):
                if possible_solution[index] == char:
                    to_remove.add(possible_solution)
                    break
                if char not in possible_solution:
                    to_remove.add(possible_solution)
                    break

    p_next = [i for i in p_i_1 if i not in to_remove]
        
    return p_next

def calculate_possible_responses(gk, pi):
    responses = []
    for solution in pi:
        responses.append(get_response(pi, gk, solution))
    responses, counts = np.unique(responses, return_counts=True, axis=0)
    probs = counts/sum(counts)
    return responses, probs

def compute_Im(pi,ci):
    return np.log2(len(pi)/len(ci))

def get_next_guess(pi, guessi, return_im = False, method='entropy'):
    guesses = {}
    
    if len(pi) == 1:
        return pi[0]
    
    n = 0
    times = []

    for gk in pi + guessi:

        start = time.time()
        n+=1

        im = []
        responses, probabilities = calculate_possible_responses(gk, pi)
        for index, response in enumerate(responses):
            pi1 = reduce_pool(pi, gk, response)
            if method == 'entropy':
                im.append(compute_Im(pi, pi1)*probabilities[index])
                print(gk)
                print(compute_Im(pi, pi1), probabilities[index] )
                print(compute_Im(pi, pi1)*probabilities[index])
                assert False
            else:
                im.append(compute_Im(pi, pi1))
        guesses[gk] = im
        
        end = time.time()
        t = end-start
        times.append(t)
        if (n%1000 ==0):
            t = np.mean(times)
            print(f'Took {t} seconds. About {t*(len(pi+guessi)-n)} seconds remaining', end='\r')
            times = []

        
    # Different strats
    # MaxMin
#     pprint.pprint([(k,min(v)) for k, v in guesses.items()])
    next_guess = max([(k,min(v)) for k, v in guesses.items()], key= lambda x: x[1])
    return next_guess
    
# #   AvgEntropy
#     guess_vals = [(k,sum(v)) for k, v in guesses.items()]
# #     for i in guess_vals:
# #         print(i)
#     listy = np.array(guess_vals)
#     winner_idx = np.argwhere(listy[:,1] == np.amax(listy[:,1]))
#     if len(winner_idx) > 1:
#         tie_break = []
#         for k in listy[winner_idx, 0].T[0]:
# #             print(k, guesses[k])
#             tie_break.append((k,min(guesses[k])))
# #         print(tie_break)
#         next_guess = max(tie_break, key= lambda x: x[1])
#     else:
#         next_guess = max(guess_vals, key= lambda x: x[1])
    
#     if return_im:
#         return next_guess
#     else:
#         return next_guess[0]