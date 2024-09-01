from graph import Graph
from mpl_toolkits.axes_grid1 import make_axes_locatable

from tools_for_spectrum import DataAndProcessing


# ШАБЛОНЫ ОТРИСОВКИ ГРАФИКОВ
# Очистка и подпись графика (вызывается в начале)
def cleaning_and_chart_graph(graph: Graph, x_label, y_label, title):
    graph.toolbar.home()  # Возвращаем зум в домашнюю позицию
    graph.toolbar.update()  # Очищаем стек осей (от старых x, y lim)
    # Очищаем график
    graph.axis.clear()
    # Задаем название осей
    graph.axis.set_xlabel(x_label)
    graph.axis.set_ylabel(y_label)
    # Задаем название графика
    graph.axis.set_title(title)


# Отрисовка (вызывается в конце)
def draw_graph(graph: Graph):
    # Убеждаемся, что все помещается внутри холста
    graph.figure.tight_layout()
    # Показываем новую фигуру в интерфейсе
    graph.canvas.draw()


# Класс художник. Имя холст (graph), рисует на нем данные
class Drawer:
    # Цвет графиков
    signal_color = "#ff0000"  # Красный цвет графика

    # ОТРИСОВКИ
    # (1) График исходного сигнала
    @staticmethod
    def graph_signal(
            graph: Graph,
            data_x,
            data_y
    ):

        # Очистка, подпись графика и осей (вызывается в начале)
        cleaning_and_chart_graph(
            # Объект графика
            graph=graph,
            # Название графика
            title=graph.name_graphics,
            # Подпись осей
            x_label=graph.horizontal_axis_name_data, y_label=graph.vertical_axis_name_data
        )

        # Рисуем график
        graph.axis.plot(
            data_y,
            data_x,
            color=Drawer.signal_color)

        # Отрисовка (вызывается в конце)
        draw_graph(graph)

    # (2) График спектра сигнала
    @staticmethod
    def graph_spectrum(
            graph: Graph,
            data: DataAndProcessing,
            spectrum_dots
    ):

        # Очистка, подпись графика и осей (вызывается в начале)
        cleaning_and_chart_graph(
            # Объект графика
            graph=graph,
            # Название графика
            title=graph.name_graphics,
            # Подпись осей
            x_label=graph.horizontal_axis_name_data, y_label=graph.vertical_axis_name_data
        )

        # Данных нет
        if data.spectrum.size == 0:
            return

        if data.abs_spectrum.size == 0:
            data.calculating_spectrum_abs()

        # Рисуем график
        size_signal = len(spectrum_dots)
        graph.axis.plot(
            spectrum_dots[:size_signal // 2],
            data.abs_spectrum[:size_signal // 2],
            color=Drawer.signal_color)

        # Отрисовка (вызывается в конце)
        draw_graph(graph)

    # (3) График спектрограммы
    @staticmethod
    def graph_color_spectrogram(
            graph: Graph,
            data,
            extent_data,
            logarithmic_axis=False
    ):

        # Очистка, подпись графика и осей (вызывается в начале)
        cleaning_and_chart_graph(
            # Объект графика
            graph=graph,
            # Название графика
            title=graph.name_graphics,
            # Подпись осей
            x_label=graph.horizontal_axis_name_data, y_label=graph.vertical_axis_name_data
        )

        # Если нужна логарифмическая ось
        norm_axis = "linear"
        if logarithmic_axis:
            norm_axis = "log"

        im = graph.axis.imshow(data, aspect='auto', extent=extent_data, norm=norm_axis)

        # Колор бар - это то, что находится справа от рисунка (столбец), НЕ ТРОГАТЬ
        # Если color bar нет - создаем, иначе обновляем
        if not graph.colorbar:
            divider = make_axes_locatable(graph.axis)
            cax = divider.append_axes("right", "10%", pad="3%")
            graph.colorbar = graph.figure.colorbar(im, orientation='vertical', cax=cax)
        else:
            graph.colorbar.update_normal(im)

        # Отрисовка (вызывается в конце)
        draw_graph(graph)
