#include "neuralnet.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>

namespace py = pybind11;

namespace Labs
{
    namespace Tests
    {
        void testNeuralNetwork()
        {
            std::cout << "=== Neural Network Test ===\n";
            Neural::NeuralNetwork<float> net(2, 16);  // 2 признака, 16 нейронов
            std::vector<float> sample = { 0.5f, -0.3f };

            float prob = net.forward(sample);
            int pred = net.predictClass(sample);

            std::cout << "Forward probability: " << prob << "\n";
            std::cout << "Predicted class: " << pred << "\n";
            std::cout << "Test passed!\n";
        }
    }
}

#ifdef _DEBUG
// Для отладки в Visual Studio
int main()
{
    Labs::Tests::testNeuralNetwork();
    system("pause");
    return 0;
}
#else
// Основной модуль для Python
PYBIND11_MODULE(nn_core, m)
{
    m.doc() = "Neural Network for Laboratory works 3-4";

    py::class_<Neural::NeuralNetwork<float>>(m, "NeuralNetwork")
        .def(py::init<int, int>(),
            py::arg("input_size") = 2,
            py::arg("hidden_size") = 16,
            "Создать нейронную сеть (input_size - количество признаков, hidden_size - нейронов)")

        .def("forward", &Neural::NeuralNetwork<float>::forward,
            "Прямой проход - возвращает вероятность класса 1")

        .def("predict_class", &Neural::NeuralNetwork<float>::predictClass,
            py::arg("features"),
            py::arg("threshold") = 0.5,
            "Предсказать класс (0 или 1)")

        .def("save", &Neural::NeuralNetwork<float>::save,
            "Сохранить модель в .bin файл")

        .def("load", &Neural::NeuralNetwork<float>::load,
            "Загрузить модель из .bin файла")

        .def("get_input_size", &Neural::NeuralNetwork<float>::getInputSize)
        .def("get_hidden_size", &Neural::NeuralNetwork<float>::getHiddenSize);
}
#endif
