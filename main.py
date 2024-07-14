import math
from collections import Counter, defaultdict

import nltk
from nltk.corpus import gutenberg


def init():
    # nltk.download('gutenberg')

    n = user_input("n")
    corpus_type = user_input("corpus")

    corpus = getCorpus(corpus_type)
    tokens = tokenize(corpus)
    tokens = remove_punctuation(tokens)
    grams = getGrams(n, tokens)
    model = buildModel(grams)
    convertToProbabilities(model)
    printModel(model)
    print("---------------------------------------------------")
    while True:
        usr_input = user_input("predict").lower()
        prefix = tuple(usr_input.split())
        prediction = predict(model, prefix)
        test_tokens = tokenize(usr_input)
        perplexity(model, test_tokens)
        if prediction:
            print(f"The prediction for the prefix: {prefix} is: {prediction}")
        else:
            print("No prediction :(")
        print("---------------------------------------------------")


def user_input(request_id):
    match request_id:
        case "n":
            return int(input("Dimension of n?"))
        case "corpus":
            return input("Corpus size?")
        case "predict":
            return input("For which word(s) you want to predict?")


def read_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return file.read()


def getCorpus(type):
    match type:
        case "basic":
            return "Das ist ein Beispieltext"
        case "basic+":
            return "Das ist gut und das ist schlecht"
        case "goethe":
            return read_file("werter.txt")
        case "datascience":
            return read_file("wikipedia_datascience.txt")
        #case "gutenberg":
           # return gutenberg.raw('austen-emma.txt')


def tokenize(text):
    return nltk.word_tokenize(text.lower())


def remove_punctuation(tokens):
    return [word for word in tokens if word.isalpha()]


def getGram(gram, tokens):
    return list(nltk.ngrams(tokens, gram))


def getGrams(n, tokens):
    grams = [[] for _ in range(n)]
    for i in range(1, n + 1):
        grams[i - 1] = getGram(i, tokens)
    return grams


def getGramCount(gram):
    return Counter(gram)


def buildModel(grams):
    model = defaultdict(lambda: defaultdict(lambda: 0))
    for gram_list in grams:
        for gram in gram_list:
            prefix = gram[:-1]
            suffix = gram[-1]
            if prefix:  # Keine () -> Wort predictions
                model[prefix][suffix] += 1
    return model


def printModel(model):
    for prefix in model:
        for suffix in model[prefix]:
            print(f"{prefix} -> {suffix}: {model[prefix][suffix]}")


def convertToProbabilities(model):
    for prefix in model:
        total_count = sum(model[prefix].values())

        for suffix in model[prefix]:
            model[prefix][suffix] /= total_count


def predict(model, prefix):
    if prefix in model:
        possible_words = model[prefix]
        return max(possible_words, key=possible_words.get)


def perplexity(model, test_data_tokens):
    grams = getGrams(len(test_data_tokens), test_data_tokens)
    accumulator = 0.0

    for gram_list in grams:
        for gram in gram_list:
            prefix = tuple(gram[:-1])
            suffix = gram[-1]
            if len(prefix) > 0:
                if prefix in model and suffix in model[prefix]:
                    propabilitie = model[prefix][suffix]
                    accumulator += math.log(propabilitie)
                else:
                    accumulator += math.log(1e-10)

    perplexity_result = math.exp(accumulator / -len(test_data_tokens))
    print(f"Perplexity: {perplexity_result}")


if __name__ == '__main__':
    init()
