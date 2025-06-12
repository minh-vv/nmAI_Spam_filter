# performance.py

class ModelAccuracy:
    NAIVE_BAYES = 0.8679
    RANDOM_FOREST = 0.9750
    LOGISTIC_REG = 0.9708

    @classmethod
    def total_accuracy(cls):
        return sum([cls.NAIVE_BAYES, cls.RANDOM_FOREST, cls.LOGISTIC_REG])
