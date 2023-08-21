import numpy as np

import utils


class GeneticAlgoritm:
    def __init__(self):
        self.population_size = 100
        self.nsurv = 20
        self.nnew = self.population_size - self.nsurv
        self.epochs = 200
        self.mut = 0.25
        pass


    def cross_point(self, curr_popul: np.array)->np.array:
        """Скрещивание двух родителей

        Args:
            curr_popul (np.array): упорядоченная популяция

        Returns:
            np.array: потомок
        """
        try:
            bots = curr_popul[np.random.randint(0, self.nsurv - 1, size=2)]
            max_indexes = curr_popul.max()
            indexes = np.random.choice(max_indexes, size=curr_popul.shape[1])
            for i in range(max_indexes):
                bots[i, np.where(indexes!=i)[0]] = 0
            return bots.sum(axis=0)
        except Exception as e:
            return curr_popul[0]



class GeneticRemovingZeros (GeneticAlgoritm):    
    def __init__(self, checkpoint_count:int) -> None:
        """Инициализация объекта генетического алгоритма для удаления нулей из временного поля

        Args:
            checkpoint_count (int): Количество контрольных точек
        """
        super().__init__()
        self.bot_length = checkpoint_count*2
        self.popul = np.random.choice(2, size=(self.population_size, self.bot_length))


    def process(self, field:np.array)->np.array:
        """Запуск генетического алгоритма

        Args:
            field (np.array): временная карта

        Returns:
            np.array: массив с исключающими колонками и столбцами
        """
        for _ in range(self.epochs):
            validation = np.zeros(shape=self.population_size).astype(np.int16)
            for i, bot in enumerate(self.popul):
                cut_field = utils.cut_matrix(field, bot)
                if np.where(cut_field==0)[0].size == 0:
                    validation[i] = cut_field.size - np.abs(bot.sum() - self.bot_length//2) * 10
                else:
                    validation[i] = -cut_field.size - np.abs(bot.sum() - self.bot_length//2) * 10
            new_popul = self.popul[validation.argsort()[::-1]]
            for i in range(self.nnew):
                new_popul[self.nsurv+i] = self.cross_point(new_popul)
                if np.random.random() < self.mut:
                    new_popul[self.nsurv+i] = np.random.choice(2, size=self.bot_length)
            self.popul = new_popul
        return self.popul[0], validation[0]
