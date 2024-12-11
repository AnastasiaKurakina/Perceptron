import numpy as np
import matplotlib.pyplot as plt

# Установка начального значения для генератора случайных чисел для воспроизводимости
np.random.seed(0)

# Генерация выборки из 250 случайных точек в двумерном пространстве
n_samples = 250
X = np.random.rand(n_samples, 2)  # Случайные координаты (x, y)

# Определение меток классов на основе заданного условия
Y = (((7 * X[:, 0] - 3)) + (X[:, 1] - 0.3) < 0.5).astype(int)  # Преобразуем в бинарный формат (0 и 1)

class Perceptron:
    def __init__(self, learning_rate=0.1, n_iterations=1000):
        # Инициализация параметров перцептрона
        self.learning_rate = learning_rate  # Скорость обучения
        self.n_iterations = n_iterations    # Количество итераций обучения
        self.weights = np.zeros(2)          # Веса модели инициализируются нулями
        self.bias = 0                        # Смещение модели инициализируется нулем

    def fit(self, X, y):
        # Метод для обучения перцептрона на данных X с метками y
        n_samples, n_features = X.shape      # Получаем количество образцов и признаков

        for _ in range(self.n_iterations):    # Обучение в течение заданного количества итераций
            for idx, x_i in enumerate(X):     # Проходим по каждому образцу данных
                linear_output = np.dot(x_i, self.weights) + self.bias  # Линейная комбинация входов и весов
                y_predicted = self.activation_function(linear_output)   # Применяем активационную функцию

                # Обновление весов и смещения на основе ошибки предсказания
                update = self.learning_rate * (y[idx] - y_predicted)
                self.weights += update * x_i   # Обновление весов
                self.bias += update              # Обновление смещения

    def activation_function(self, x):
        # Пороговая активационная функция (step function)
        return np.where(x >= 0, 1, 0)         # Возвращает 1, если x >= 0, иначе 0

    def predict(self, X):
        # Метод для предсказания классов на основе входных данных X
        linear_output = np.dot(X, self.weights) + self.bias  # Линейная комбинация входов и весов
        return self.activation_function(linear_output)       # Применяем активационную функцию

# Функция для визуализации данных и разделяющей прямой с подписями
def plot_decision_boundary(X, y, model, num_samples):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k', s=20)  # Отображение точек данных
    
    # Определение границ для построения разделяющей прямой
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))  
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])   # Предсказание классов для каждой точки сетки
    Z = Z.reshape(xx.shape)                             # Преобразование предсказаний в форму сетки
    
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')   # Отображение разделяющей прямой
    
    if model.weights[1] != 0:  
        slope = -model.weights[0] / model.weights[1]
        intercept = -model.bias / model.weights[1]
        
        x_line = np.linspace(x_min, x_max)
        y_line = slope * x_line + intercept
        
        plt.plot(x_line, y_line, color='black', label='Decision Boundary')   # Отображение линии разделения
    
    plt.xlabel('Feature 1')                               # Подпись оси X
    plt.ylabel('Feature 2')                               # Подпись оси Y
    plt.title(f'Decision Boundary of Perceptron\n'
              f'Number of Samples: {num_samples}, '
              f'Final Weights: {model.weights}')         # Заголовок графика с подписями (только конечные веса)
    plt.legend()                                         # Легенда графика
    plt.show()                                           # Показать график

if __name__ == "__main__":
    # ###############################################################
    # Основной блок программы, который выполняется при запуске скрипта.
    # Здесь создается экземпляр перцептрона, обучается модель,
    # визуализируются результаты и проводятся тесты с различными начальными весами.
    # ###############################################################

    # Создание экземпляра класса Perceptron с заданной скоростью обучения и количеством итераций
    perceptron = Perceptron(learning_rate=0.1, n_iterations=1000)

    # Обучение перцептрона на всех сгенерированных данных (X, Y)
    perceptron.fit(X, Y)

    # Визуализация разделяющей прямой и данных для всех точек (250)
    plot_decision_boundary(X, Y, perceptron, num_samples=250)

    # Вывод финальных весов и смещения после обучения модели в консоль
    print("Weights:", perceptron.weights)  # Печать финальных весов
    print("Bias:", perceptron.bias)          # Печать финального смещения

    # Проверка, чтобы избежать деления на ноль при вычислении уравнения прямой
    if perceptron.weights[1] != 0: 
        # Вычисление наклона (slope) разделяющей прямой
        slope = -perceptron.weights[0] / perceptron.weights[1]
        # Вычисление пересечения (intercept) с осью Y
        intercept = -perceptron.bias / perceptron.weights[1]
        
        # Вывод уравнения разделяющей прямой в консоль
        print(f"Equation of the decision boundary: y = {slope:.2f} * x + {intercept:.2f}")
    else:
        print("The decision boundary is vertical.")  # Если прямая вертикальна

    # ###############################################################
    # Генерация меньшего множества данных (от 10 до 15 точек).
    # ###############################################################

    n_samples_small = np.random.randint(10, 16)  # Случайное количество точек от 10 до 15
    X_small = np.random.rand(n_samples_small, 2)   # Случайные координаты для меньшего множества
    Y_small = (((7 * X_small[:, 0] - 3)) + (X_small[:, 1] - 0.3) < 0.5).astype(int)  # Определение меток классов

    # ###############################################################
    # Функция для тестирования перцептрона с различными начальными значениями весов 
    # и смещения на меньшем множестве.
    #
    # Тесты:
    #
    # Тест №1: Нулевые начальные веса и смещение.
    #
    # Тест №2: Случайные начальные веса от -1 до +1.
    #
    # Тест №3: Пользовательские начальные веса [0.5, -0.5] и смещение -0.5.
    #
    # ###############################################################

    def test_perceptron_with_initial_conditions(X_test, Y_test):
        initial_conditions_list = [
            (np.zeros(2), 0),                       # Нулевые начальные веса и смещение
            (np.random.rand(2) * (np.random.choice([-1, 1])), np.random.rand() * (np.random.choice([-1, 1]))),   # Случайные начальные веса и смещение от -1 до +1 
            ([0.5, -0.5], -0.5)                    # Пример пользовательских начальных весов и смещения 
        ]
        
        for i, initial_weights in enumerate(initial_conditions_list):
            perceptron_test = Perceptron(learning_rate=0.1, n_iterations=1000)
            
            # Вывод информации о текущем тесте в консоль
            print(f"\nTesting #{i+1} with initial weights: {initial_weights[0]}, initial bias: {initial_weights[1]}")
            
            # Установка начальных весов и смещения для текущего экземпляра перцептрона
            perceptron_test.weights = initial_weights[0] if isinstance(initial_weights[0], np.ndarray) else np.array(initial_weights[0])
            perceptron_test.bias = initial_weights[1]
            
            # Обучение перцептрона на тестовых данных
            perceptron_test.fit(X_test, Y_test)
            
            # Визуализация разделяющей прямой и данных для текущего теста
            plot_decision_boundary(X_test, Y_test, perceptron_test,
                                   num_samples=len(X_test))

            # Вывод финальных весов и смещения после обучения модели в консоль
            print("Final Weights:", perceptron_test.weights)
            print("Final Bias:", perceptron_test.bias)

            if perceptron_test.weights[1] != 0: 
                slope = -perceptron_test.weights[0] / perceptron_test.weights[1]  # Наклон прямой
                intercept = -perceptron_test.bias / perceptron_test.weights[1]     # Пересечение с осью Y
                
                # Вывод уравнения разделяющей прямой в консоль
                print(f"Equation of the decision boundary: y = {slope:.2f} * x + {intercept:.2f}")
            else:
                print("The decision boundary is vertical.")  # Если прямая вертикальна

    test_perceptron_with_initial_conditions(X_small, Y_small)  # Запуск функции тестирования на меньшем множестве

