import numpy as np


class SinParameters:
    """DataClass для параметров гармоники (sin)"""
    def __init__(self, amplitude: float, frequency: float, phase: float):
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase


def get_period_fm2(frequency: float) -> float:
    """Перевод частоты в период"""
    return 1 / frequency


def sin_count(amplitude: float, frequency: float, phase: float, time: float):
    """Получение отсчета синусоиды"""
    # Фаза задается в градусах (ниже перевод из радиан в градусы)
    # A * sin(2 * pi * w * t + (phi + FM2))
    return amplitude * np.sin(2. * np.pi * frequency * time + phase)


def generation_harmonic_signal_fm2(list_harmonics: list[SinParameters], array_time: np.array, number_counts: int,
                                   sampling_rate: int, modulation_frequency: float = 20.0) -> list[float]:
    """Функция генерации сигнала из нескольких гармоник с FM2 модуляцией"""
    # Пустой сигнал необходимой длинны
    signal = [0] * number_counts

    # Модуляция
    phase_change_step = get_period_fm2(modulation_frequency)
    phase_period = 2 * phase_change_step

    # Перебор и расчет отсчетов времени
    for time in range(number_counts):
        # Добавка к фазе
        phase_additive = 0

        # Период смены фазы
        if array_time[time] % phase_period >= phase_change_step:
            phase_additive = np.pi

        # Перебор гармоник
        for harmonic in list_harmonics:
            # Сложение гармоник
            signal[time] += sin_count(
                harmonic.amplitude, harmonic.frequency, harmonic.phase + phase_additive, time / sampling_rate)

    # Возвращаем сгенерированный сигнал
    return signal


def zero_signal_boundaries(start_signal: float, end_signal: float, signal: list[float], array_time: np.array):
    """Функция обнуления участков вне заданного диапазона в сигнале"""
    maximum_time = array_time[-1]

    # Проверка корректности границ диапазона
    if (start_signal < end_signal) and (end_signal < maximum_time) and start_signal > 0 and end_signal > 0:
        for time in range(len(signal)):
            if array_time[time] < start_signal or array_time[time] > end_signal:
                signal[time] = 0.0
