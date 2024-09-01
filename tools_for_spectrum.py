import spectrum
import numpy as np


class DataAndProcessing:
    """DataAndProcessing для обработки спектра"""
    def __init__(self, signal_noise):
        self.signal_noise = signal_noise
        self.spectrum = np.array([])
        self.abs_spectrum = np.array([])
        self.name_spectrum = None

    # (1.1) Построение спектра сигнала с помощью FFT (быстрого преобразования Фурье)
    def calculating_spectrum_fft(self) -> np.array:
        """Построение спектра сигнала с помощью БПФ"""
        if self.signal_noise is None:
            return

        # Рассчитываем спектр, используя БПФ
        self.spectrum = np.fft.fft(self.signal_noise)[1:]
        self.name_spectrum = 'fft'

    # (1.2) Построение спектра сигнала с помощью AKF (автокорреляционной функции)
    def calculating_spectrum_akf(self) -> np.array:
        """Построение спектра сигнала с помощью АКФ"""
        if self.signal_noise is None:
            return

        # Считаем автокорреляционную функцию
        autocorrelation_function = np.correlate(self.signal_noise, self.signal_noise, mode="full")
        spectrum_akf = autocorrelation_function[len(autocorrelation_function) // 2 + 1:]
        length_spectrum_akf = len(spectrum_akf)
        # Берем 1/8 часть от длины, остальное заполняем нулями
        spectrum_akf = np.concatenate((spectrum_akf[:length_spectrum_akf // 8],
                                       [0] * (length_spectrum_akf - (length_spectrum_akf // 8))))
        self.spectrum = np.fft.fft(spectrum_akf)
        self.name_spectrum = 'akf'

    # (1.3) Построение спектра сигнала с помощью АR-модели (авторегрессионной модели)
    def calculating_spectrum_ar_model(self, order_model: int) -> np.array:
        """Построение спектра сигнала с помощью АR-модели"""
        if self.signal_noise is None:
            return

        # Рассчитываем коэффициенты AR модели
        ar_coefficients, noise_dispersion, reflection_coefficients = spectrum.arburg(self.signal_noise, order_model)
        spectrum_ar_model = spectrum.arma2psd(ar_coefficients, NFFT=len(self.signal_noise))
        self.spectrum = spectrum_ar_model

    # (1.4) Построение спектра сигнала с помощью MMD (метода минимальной дисперсии)
    def calculating_spectrum_mmd(self, order_model: int, sampling_rate: int) -> np.array:
        """Построение спектра сигнала с помощью ММД"""
        if self.signal_noise is None:
            return

        spectrum_mmd = spectrum.pminvar(self.signal_noise, order_model, NFFT=len(self.signal_noise),
                                        sampling=sampling_rate)
        self.spectrum = spectrum_mmd.psd

    # (1.5) Построение спектра сигнала с помощью MME (метода максимума энтропии)
    def calculating_spectrum_mme(self, order_model: int, sampling_rate: int) -> np.array:
        """Построение спектра сигнала с помощью ММЭ"""
        if self.signal_noise is None:
            return

        spectrum_mme = spectrum.pburg(self.signal_noise, order_model, NFFT=len(self.signal_noise),
                                      sampling=sampling_rate)
        self.spectrum = spectrum_mme.psd

    # Функция для модуля спектра
    def calculating_spectrum_abs(self) -> np.array:
        """Функция для модуля спектра"""
        if self.spectrum is None:
            return

        self.abs_spectrum = abs(self.spectrum)
