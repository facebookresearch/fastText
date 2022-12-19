/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <args.h>
#include <fasttext.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <real.h>
#include <vector.h>
#include <cmath>
#include <iterator>
#include <sstream>
#include <stdexcept>

using namespace pybind11::literals;
namespace py = pybind11;

py::str castToPythonString(const std::string& s, const char* onUnicodeError) {
  PyObject* handle = PyUnicode_DecodeUTF8(s.data(), s.length(), onUnicodeError);
  if (!handle) {
    throw py::error_already_set();
  }

  // py::str's constructor from a PyObject assumes the string has been encoded
  // for python 2 and not encoded for python 3 :
  // https://github.com/pybind/pybind11/blob/ccbe68b084806dece5863437a7dc93de20bd9b15/include/pybind11/pytypes.h#L930
#if PY_MAJOR_VERSION < 3
  PyObject* handle_encoded =
      PyUnicode_AsEncodedString(handle, "utf-8", onUnicodeError);
  Py_DECREF(handle);
  handle = handle_encoded;
#endif

  py::str handle_str = py::str(handle);
  Py_DECREF(handle);
  return handle_str;
}

std::vector<std::pair<fasttext::real, py::str>> castToPythonString(
    const std::vector<std::pair<fasttext::real, std::string>>& predictions,
    const char* onUnicodeError) {
  std::vector<std::pair<fasttext::real, py::str>> transformedPredictions;

  for (const auto& prediction : predictions) {
    transformedPredictions.emplace_back(
        prediction.first,
        castToPythonString(prediction.second, onUnicodeError));
  }

  return transformedPredictions;
}

PYBIND11_MODULE(fasttext_pybind, m) {
  py::class_<fasttext::FastText>(m, "fasttext")
      .def(py::init<>())
      .def(
          "loadModel",
          [](fasttext::FastText& m, std::string s) { m.loadModel(s); })
      .def(
          "predict",
          // NOTE: text needs to end in a newline
          // to exactly mimic the behavior of the cli
          [](fasttext::FastText& m,
             const std::string text,
             int32_t k,
             fasttext::real threshold,
             const char* onUnicodeError) {
            std::stringstream ioss(text);
            std::vector<std::pair<fasttext::real, std::string>> predictions;
            m.predictLine(ioss, predictions, k, threshold);

            return castToPythonString(predictions, onUnicodeError);
          })
      .def(
          "multilinePredict",
          // NOTE: text needs to end in a newline
          // to exactly mimic the behavior of the cli
          [](fasttext::FastText& m,
             const std::vector<std::string>& lines,
             int32_t k,
             fasttext::real threshold,
             const char* onUnicodeError) {
            std::vector<std::vector<py::str>> allLabels;
            std::vector<std::pair<fasttext::real, std::string>> predictions;

            for (const std::string& text : lines) {
              std::stringstream ioss(text);
              m.predictLine(ioss, predictions, k, threshold);
              std::vector<fasttext::real> probabilities;
              std::vector<py::str> labels;

              for (const auto& prediction : predictions) {
                probabilities.push_back(prediction.first);
                labels.push_back(
                    castToPythonString(prediction.second, onUnicodeError));
              }
              allLabels.push_back(labels);
            }

            return allLabels;
          });
}
