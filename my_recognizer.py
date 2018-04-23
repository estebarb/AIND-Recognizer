import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    
    # TODO implement the recognizer
    # return probabilities, guesses
    def try_or_default(fn, *args, default=float("-Inf")):
        try:
            return fn(*args)
        except:
            return default
    for _, (X, lengths) in test_set.get_all_Xlengths().items():
        probability_dict = {
            word: try_or_default(model.score, X, lengths)
            for word, model in models.items()
            }
        probabilities.append(probability_dict)
        guesses.append(max(probability_dict, key=probability_dict.get))
    return probabilities, guesses
    
    
