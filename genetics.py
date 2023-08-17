import numpy as np

import utils

class GeneticRemovingZeros:
    population_size = 250
    nsurv = 50
    nnew = population_size - nsurv
    epochs = 500
    mut = 0.25

    def __init__(self, checkpoint_count:int) -> None:
        """Инициализация объекта генетического алгоритма для удаления нулей из временного поля

        Args:
            checkpoint_count (int): Количество контрольных точек
        """
        self.bot_length = checkpoint_count*2
        self.popul = np.random.choice(2, size=(self.__class__.population_size, self.bot_length))

  
    def cross_point(self, curr_popul: np.array)->np.array:
        """Скрещивание двух родителей

        Args:
            curr_popul (np.array): упорядоченная популяция

        Returns:
            np.array: потомок
        """
        bots = curr_popul[np.random.randint(0, self.__class__.nsurv - 1, size=2)]
        indexes = np.random.choice(2, size=10)
        bots[0,np.where(indexes==1)[0]] = 0
        bots[1,np.where(indexes==0)[0]] = 0
        return bots.sum(axis=0)

    
    def process(self, field:np.array)->np.array:
        """Запуск генетического алгоритма

        Args:
            field (np.array): временная карта

        Returns:
            np.array: массив с исключающими колонками и столбцами
        """
        for _ in range(self.__class__.epochs):
            validation = np.zeros(shape=self.__class__.population_size).astype(np.int16)
            for i, bot in enumerate(self.popul):
                cut_field = utils.cut_matrix(field, bot)
                if np.where(cut_field==0)[0].size == 0:
                    validation[i] = cut_field.size
                else:
                    validation[i] = -cut_field.size
            new_popul = self.popul[validation.argsort()[::-1]]
            for i in range(self.__class__.nnew):
                new_popul[self.__class__.nsurv+i] = self.cross_point(new_popul)
                if np.random.random() < self.__class__.mut:
                    new_popul[self.__class__.nsurv+i] = np.random.choice(2, size=self.bot_length)
            self.popul = new_popul
        return self.popul[0]
