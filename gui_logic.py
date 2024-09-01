import math
import time
import random
import matplotlib
import numpy as np
from scipy.io import wavfile
from scipy.fft import fftfreq
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog

from graph import Graph
from gui import Ui_Dialog
from drawer import Drawer as drawer
from tools_for_spectrum import DataAndProcessing
from data_processing.signal_generation import SinParameters, generation_harmonic_signal_fm2, zero_signal_boundaries

matplotlib.use('TkAgg')


# Функция для шума (нормальное распределение по Гауссу)
def uniform_distribution() -> float:
    """Функция для шума"""
    repeat = 12
    val = 0
    for i in range(repeat):
        val += random.random()  # значение от 0.0 до 1.0

    return val / repeat


# КЛАСС АЛГОРИТМА ПРИЛОЖЕНИЯ
class GuiProgram(Ui_Dialog):

    def __init__(self, dialog):
        # Создаем окно
        Ui_Dialog.__init__(self)
        # Дополнительные функции окна
        dialog.setWindowFlags(  # Передаем флаги создания окна
            Qt.WindowCloseButtonHint |  # Закрытие
            Qt.WindowMaximizeButtonHint |  # Во весь экран (развернуть)
            Qt.WindowMinimizeButtonHint  # Свернуть
        )
        self.setupUi(dialog)  # Устанавливаем пользовательский интерфейс
        # ПОЛЯ КЛАССА
        # Параметры 1 графика - Исходный сигнал
        self.graph_1 = Graph(
            layout=self.layout_plot,
            widget=self.widget_plot,
            name_graphics="График №1. Исходный сигнал",
            horizontal_axis_name_data="Время (t) [c]",
            vertical_axis_name_data="Амплитуда (A) [отн. ед.]"
        )
        # Параметры 2 графика - Исходный сигнал с шумом
        self.graph_2 = Graph(
            layout=self.layout_plot_2,
            widget=self.widget_plot_2,
            name_graphics="График №1. Исходный сигнал c шумом",
            horizontal_axis_name_data="Время (t) [c]",
            vertical_axis_name_data="Амплитуда (A) [отн. ед.]"
        )
        # Параметры 3 графика - Спектр сигнала
        self.graph_3 = Graph(
            layout=self.layout_plot_3,
            widget=self.widget_plot_3,
            name_graphics="График №3. Спектр сигнала",
            horizontal_axis_name_data="Частота (f) [Гц]",
            vertical_axis_name_data="Амплитуда (A) [отн. ед.]"
        )
        # Параметры 4 графика - Спектрограмма сигнала
        self.graph_4 = Graph(
            layout=self.layout_plot_4,
            widget=self.widget_plot_4,
            name_graphics="График №2. Спектрограмма",
            horizontal_axis_name_data="Время (t) [с]",
            vertical_axis_name_data="Частота (f) [отн. ед.]"
        )

        self.data_and_processing: DataAndProcessing | None = None

        # Этапы обработки сигнала
        self.original_signal = None
        # Точки для сигнала
        self.signal_dots = None
        # Размер сигнала
        self.size_signal = None

        # Длительность сигнала
        self.maximum_time = None
        # Частота дискретизации
        self.sampling_rate = None

        # Сигнал с шумом
        self.noise_signal = None
        # Точки для спектра
        self.spectrum_dots = None

        # Данные спектрограммы
        self.data_spectrogram = None
        self.chart_counts_k = None
        self.graph_k = None

        # Алгоритм обратки
        # Генерация сигнала
        self.pushButton_generate_signal.clicked.connect(self.generation_signal)
        # Загрузка аудиофайла
        self.pushButton_downloading_an_audio_file.clicked.connect(self.downloading_an_audio_file)
        # Загрузка данных из файла
        self.pushButton_uploading_a_file.clicked.connect(self.uploading_a_file)
        # Добавление шума к исходному сигналу
        self.pushButton_display_noise.clicked.connect(self.add_noise_in_decibels)
        # Построение спектра сигнала
        self.pushButton_building_spectrum.clicked.connect(self.calculate_spectrum_signal_noise)
        # Построение спектрограммы сигнала
        self.pushButton_building_spectrogram.clicked.connect(self.calculate_spectrogram)

    # АЛГОРИТМ РАБОТЫ ПРОГРАММЫ
    # (1а.1) Построение сигнала
    def generation_signal(self):
        """Генерация сигнала"""
        # Запрашиваем параметры синусоид
        # Первая синусоида
        amplitude_1 = float(self.lineEdit_amplitude_1.text())
        frequency_1 = float(self.lineEdit_frequency_1.text())
        phase_1 = float(self.lineEdit_phase_1.text()) / 180 * np.pi  # перевод в градусы
        harmonic_1 = SinParameters(amplitude_1, frequency_1, phase_1)
        # Вторая синусоида
        amplitude_2 = float(self.lineEdit_amplitude_2.text())
        frequency_2 = float(self.lineEdit_frequency_2.text())
        phase_2 = float(self.lineEdit_phase_2.text()) / 180 * np.pi  # перевод в градусы
        harmonic_2 = SinParameters(amplitude_2, frequency_2, phase_2)
        # Третья синусоида
        amplitude_3 = float(self.lineEdit_amplitude_3.text())
        frequency_3 = float(self.lineEdit_frequency_3.text())
        phase_3 = float(self.lineEdit_phase_3.text()) / 180 * np.pi  # перевод в градусы
        harmonic_3 = SinParameters(amplitude_3, frequency_3, phase_3)

        # Запрашиваем параметр для FM-2 модуляции
        modulation_frequency = float(self.lineEdit_modulation_frequency.text())
        # Запрашиваем частоту дискретизации (fd)
        self.sampling_rate = int(self.lineEdit_sampling_rate.text())
        # Запрашиваем максимальное время (t_max)
        self.maximum_time = float(self.lineEdit_maximum_time.text())
        # Считаем количество отсчетов (N = t_max * fd)
        number_counts = int(self.sampling_rate * self.maximum_time)
        # Создаем пустой сигнал размером N
        self.original_signal = np.zeros(number_counts)
        # Шаг по времени (step (или дельта t) = 1 / fd)
        step_time = 1 / self.sampling_rate
        # Отсчеты для отображения оси времени
        self.signal_dots = np.arange(0, self.maximum_time, step_time)

        # Генерации гармонического сигнала с FM2 модуляцией
        self.original_signal = generation_harmonic_signal_fm2(
            list_harmonics=[harmonic_1, harmonic_2, harmonic_3],
            array_time=self.signal_dots,
            number_counts=number_counts,
            sampling_rate=self.sampling_rate,
            modulation_frequency=modulation_frequency
        )

        # Обнуление участков вне заданного диапазона в сигнале
        start_signal = float(self.lineEdit_start_signal.text())
        end_signal = float(self.lineEdit_end_signal.text())
        zero_signal_boundaries(start_signal, end_signal, self.original_signal, self.signal_dots)
        # Отображаем исходный сигнал
        drawer.graph_signal(self.graph_1, self.original_signal, self.signal_dots)

    # (1а.2) Добавление шума в децибелах (дБ)
    def add_noise_in_decibels(self):
        """Добавление шума (в дБ) к сигналу"""
        # Нет исходного сигнала - сброс
        if self.original_signal is None:
            return

        self.size_signal = len(self.original_signal)
        # Создаем массив отсчетов шума равный размеру сигнала
        noise_counting = np.zeros(self.size_signal)

        # Считаем энергию шума
        energy_noise = 0
        for j in range(self.size_signal):
            val = uniform_distribution()
            # Записываем отсчет шума
            noise_counting[j] = val
            energy_noise += val * val

        # Считаем энергию исходного сигнала
        energy_signal = 0
        for i in range(self.size_signal):
            energy_signal += self.original_signal[i] * self.original_signal[i]

        # Запрашиваем шум в дБ
        noise_decibels = float(self.lineEdit_noise.text())
        # Считаем коэффициент/множитель шума: sqrt(10^(x/10) * (E_signal / E_noise)), x - с экрана
        noise_coefficient = math.sqrt(pow(10, (noise_decibels / 20)) * (energy_signal / energy_noise))
        # Копируем исходный сигнал
        self.noise_signal = self.original_signal.copy()
        # К отсчетам исходного сигнала добавляем отсчеты шума
        for k in range(self.size_signal):
            self.noise_signal[k] += noise_coefficient * noise_counting[k]

        self.data_and_processing = DataAndProcessing(self.noise_signal)
        # Отображаем итог
        drawer.graph_signal(self.graph_2, self.noise_signal, self.signal_dots)

    # (1б) Загрузка аудиофайла
    def downloading_an_audio_file(self):
        """Загрузка аудиофайла"""
        # Вызов окна выбора файла
        filename, filetype = QFileDialog.getOpenFileName(None,
                                                         "Выбрать файл изображения",
                                                         ".",
                                                         "All Files(*)")
        # Загружаем аудиофайл
        self.sampling_rate, data = wavfile.read(filename)
        self.noise_signal = data[:, 0]
        self.size_signal = len(self.noise_signal)
        self.maximum_time = self.size_signal / self.sampling_rate
        # Временная ось
        self.signal_dots = np.linspace(0, self.maximum_time, num=self.size_signal)

        # Вывод информации об аудиофайле
        print(f"{self.size_signal} - количество отсчётов, {self.maximum_time} - длительность, "
              f"{self.sampling_rate} - частота дискретизации")
        self.data_and_processing = DataAndProcessing(self.noise_signal)
        # Отображаем итог
        drawer.graph_signal(self.graph_1, self.noise_signal, self.signal_dots)

    # (1в) Загрузка данных из файла
    def uploading_a_file(self):
        """Загрузка данных из файла"""
        # Вызов окна выбора файла
        filename, filetype = QFileDialog.getOpenFileName(None,
                                                         "Выбрать файл изображения",
                                                         ".",
                                                         "All Files(*)")
        # Считываем данные из файла
        with open(filename, 'r') as file:
            data_file = file.read().strip().split()
        # Преобразовываем строку данных в массив чисел
        data = np.array([float(x) for x in data_file])
        # Первая половина файла - сам сигнал, вторая - временная ось
        self.noise_signal = data[len(data) // 2:]
        self.signal_dots = data[:len(data) // 2]

        self.size_signal = len(self.noise_signal)
        # Запрашиваем частоту дискретизации (fd)
        self.sampling_rate = int(self.lineEdit_sampling_rate.text())
        self.maximum_time = self.size_signal / self.sampling_rate

        self.data_and_processing = DataAndProcessing(self.noise_signal)
        # Отображаем итог
        drawer.graph_signal(self.graph_1, self.noise_signal, self.signal_dots)

    # (2) Построение спектра сигнала с помощью линейных и нелинейных методов
    def calculate_spectrum_signal_noise(self):
        """Построение спектра сигнала"""
        if not self.data_and_processing:
            return

        # Считаем спектр
        # Выбран метод FFT
        if self.radioButton_FFT.isChecked():
            self.data_and_processing.calculating_spectrum_fft()
            # Отсчеты для отображения оси частот
            self.spectrum_dots = fftfreq(self.size_signal, 1 / self.sampling_rate)
        # Выбран метод AKF
        elif self.radioButton_AKF.isChecked():
            self.data_and_processing.calculating_spectrum_akf()
            # Отсчеты для отображения оси частот
            self.spectrum_dots = np.array(
                [i * self.sampling_rate / self.size_signal for i in np.arange(self.size_signal)])
        # Выбран метод АR-модель
        elif self.radioButton_AR.isChecked():
            order_model = int(self.lineEdit_order_model.text())
            self.data_and_processing.calculating_spectrum_ar_model(order_model)
            # Отсчеты для отображения оси частот
            self.spectrum_dots = np.linspace(0, self.sampling_rate, self.size_signal)
        # Выбран метод MMD
        elif self.radioButton_MMD.isChecked():
            order_model = int(self.lineEdit_order_model.text())
            self.data_and_processing.calculating_spectrum_mmd(order_model, self.sampling_rate)
            # Отсчеты для отображения оси частот
            self.spectrum_dots = np.linspace(0, self.sampling_rate, self.size_signal)
        # Выбран метод MME
        elif self.radioButton_MME.isChecked():
            order_model = int(self.lineEdit_order_model.text())
            self.data_and_processing.calculating_spectrum_mme(order_model, self.sampling_rate)
            # Отсчеты для отображения оси частот
            self.spectrum_dots = np.linspace(0, self.sampling_rate, self.size_signal)

        # Берем модуль спектра
        self.data_and_processing.calculating_spectrum_abs()
        # Отображаем половину спектра сигнала
        drawer.graph_spectrum(self.graph_3, self.data_and_processing, self.spectrum_dots)

    # (3) Построение спектрограммы сигнала
    def calculate_spectrogram(self):
        """Построение спектрограммы сигнала"""
        if self.noise_signal is None or not self.data_and_processing:
            return

        data = []
        x = []

        # Запрашиваем размер окна
        win_size = int(self.lineEdit_window_size.text())
        # Запрашиваем шаг окна
        win_step = int(self.lineEdit_window_step.text())

        start_time = time.time()

        # Считаем спектрограмму передвижением окна (win_size) по сигналу с шумом
        for ind_t in np.arange(0, len(self.noise_signal) - win_size, win_step):
            # Начало и конец окна в сигнале
            sub_signal = self.noise_signal[int(ind_t): int(ind_t + win_size)]
            # Расчет спектра в окне
            data.append(self.calculate_spectrum(sub_signal))
            # Отсчеты времени для оси х
            x.append(ind_t / self.sampling_rate)

        data = np.array(data)
        data = data.transpose()
        if self.radioButton_MME.isChecked() or self.radioButton_MMD.isChecked():
            data = data[::-1]
            data = data[:int(win_size // 2)]
        else:
            data = data[int(win_size // 2):]

        end_time = time.time()
        print(f'Время построения спектрограммы: {end_time - start_time}')

        self.data_spectrogram = data
        self.chart_counts_k = np.array(x)
        extent_data = [0, 50, 0, 0.5]
        drawer.graph_color_spectrogram(self.graph_4, abs(data), extent_data=extent_data, logarithmic_axis=False)

    # (3.1) Выбор метода расчета спектра для построения спектрограммы
    def calculate_spectrum(self, sub_signal):
        """Расчет спектра для спектрограммы"""
        data_and_processing: DataAndProcessing = DataAndProcessing(sub_signal)

        # Считаем спектр
        # Выбран метод FFT
        if self.radioButton_FFT.isChecked():
            data_and_processing.calculating_spectrum_fft()
        # Выбран метод AKF
        elif self.radioButton_AKF.isChecked():
            data_and_processing.calculating_spectrum_akf()
        # Выбран метод АR-модель
        elif self.radioButton_AR.isChecked():
            order_model = int(self.lineEdit_order_model.text())
            data_and_processing.calculating_spectrum_ar_model(order_model)
        # Выбран метод MMD
        elif self.radioButton_MMD.isChecked():
            order_model = int(self.lineEdit_order_model.text())
            data_and_processing.calculating_spectrum_mmd(order_model, self.sampling_rate)
        # Выбран метод MME
        elif self.radioButton_MME.isChecked():
            order_model = int(self.lineEdit_order_model.text())
            data_and_processing.calculating_spectrum_mme(order_model, self.sampling_rate)

        # Возвращаем посчитанный спектр
        return data_and_processing.spectrum
