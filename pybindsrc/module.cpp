/**
 * @file module.cpp
 *
 * This is part of the DUNE DAQ Software Suite, copyright 2020.
 * Licensing/copyright details are in the COPYING file that you should have
 * received with this code.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iomanip>

#include "dtpemulator/TPGenerator.hpp"

namespace py = pybind11;
using namespace pybind11::literals;

template<typename T, typename U> constexpr size_t offset_of(U T::*member)
{
    return (char*)&((T*)nullptr->*member) - (char*)nullptr;
}

namespace dunedaq
{
  namespace dtpemulator
  {
    namespace python
    {

      PYBIND11_MODULE(_daq_dtpemulator_py, m)
      {
        m.doc() = "c++ implementation of the dunedaq dtp emulator modules"; // optional module docstring

        py::class_<TPGenerator>(m, "TPGenerator")
            .def(py::init<const std::string, const unsigned int, const unsigned int>())
            .def("pedestal_subtraction", &TPGenerator::pedestal_subtraction, py::arg("adcs"), py::arg("ini_median"), py::arg("ini_accum"), py::arg("limit") = 10)
            .def("fir_filter", &TPGenerator::fir_filter, py::arg("adcs"))
            .def("hit_finder", &TPGenerator::hit_finder, py::arg("adcs"), py::arg("tov_min") = 4)
            ;

      }

    } // namespace python
  }   // namespace dtpemulator
} // namespace dunedaq
