/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <args.h>
#include <autotune.h>
#include <densematrix.h>
#include <fasttext.h>
#include <pybind11/numpy.h>
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

std::pair<std::vector<py::str>, std::vector<py::str>> getLineText(
    fasttext::FastText& m,
    const std::string text,
    const char* onUnicodeError) {
  std::shared_ptr<const fasttext::Dictionary> d = m.getDictionary();
  std::stringstream ioss(text);
  std::string token;
  std::vector<py::str> words;
  std::vector<py::str> labels;
  while (d->readWord(ioss, token)) {
    uint32_t h = d->hash(token);
    int32_t wid = d->getId(token, h);
    fasttext::entry_type type = wid < 0 ? d->getType(token) : d->getType(wid);

    if (type == fasttext::entry_type::word) {
      words.push_back(castToPythonString(token, onUnicodeError));
      // Labels must not be OOV!
    } else if (type == fasttext::entry_type::label && wid >= 0) {
      labels.push_back(castToPythonString(token, onUnicodeError));
    }
    if (token == fasttext::Dictionary::EOS) {
      break;
}
  }
  return std::pair<std::vector<py::str>, std::vector<py::str>>(words, labels);
}

PYBIND11_MODULE(fasttext_pybind, m) {
  py::class_<fasttext::Args>(m, "args")
      .def(py::init<>())
      .def_readwrite("input", &fasttext::Args::input)
      .def_readwrite("output", &fasttext::Args::output)
      .def_readwrite("lr", &fasttext::Args::lr)
      .def_readwrite("lrUpdateRate", &fasttext::Args::lrUpdateRate)
      .def_readwrite("dim", &fasttext::Args::dim)
      .def_readwrite("ws", &fasttext::Args::ws)
      .def_readwrite("epoch", &fasttext::Args::epoch)
      .def_readwrite("minCount", &fasttext::Args::minCount)
      .def_readwrite("minCountLabel", &fasttext::Args::minCountLabel)
      .def_readwrite("neg", &fasttext::Args::neg)
      .def_readwrite("wordNgrams", &fasttext::Args::wordNgrams)
      .def_readwrite("loss", &fasttext::Args::loss)
      .def_readwrite("model", &fasttext::Args::model)
      .def_readwrite("bucket", &fasttext::Args::bucket)
      .def_readwrite("minn", &fasttext::Args::minn)
      .def_readwrite("maxn", &fasttext::Args::maxn)
      .def_readwrite("thread", &fasttext::Args::thread)
      .def_readwrite("t", &fasttext::Args::t)
      .def_readwrite("label", &fasttext::Args::label)
      .def_readwrite("verbose", &fasttext::Args::verbose)
      .def_readwrite("pretrainedVectors", &fasttext::Args::pretrainedVectors)
      .def_readwrite("saveOutput", &fasttext::Args::saveOutput)
      .def_readwrite("seed", &fasttext::Args::seed)

      .def_readwrite("qout", &fasttext::Args::qout)
      .def_readwrite("retrain", &fasttext::Args::retrain)
      .def_readwrite("qnorm", &fasttext::Args::qnorm)
      .def_readwrite("cutoff", &fasttext::Args::cutoff)
      .def_readwrite("dsub", &fasttext::Args::dsub)

      .def_readwrite(
          "autotuneValidationFile", &fasttext::Args::autotuneValidationFile)
      .def_readwrite("autotuneMetric", &fasttext::Args::autotuneMetric)
      .def_readwrite(
          "autotunePredictions", &fasttext::Args::autotunePredictions)
      .def_readwrite("autotuneDuration", &fasttext::Args::autotuneDuration)
      .def_readwrite("autotuneModelSize", &fasttext::Args::autotuneModelSize)
      .def("setManual", [](fasttext::Args& m, const std::string& argName) {
        m.setManual(argName);
      });

  py::enum_<fasttext::model_name>(m, "model_name")
      .value("cbow", fasttext::model_name::cbow)
      .value("skipgram", fasttext::model_name::sg)
      .value("supervised", fasttext::model_name::sup)
      .export_values();

  py::enum_<fasttext::loss_name>(m, "loss_name")
      .value("hs", fasttext::loss_name::hs)
      .value("ns", fasttext::loss_name::ns)
      .value("softmax", fasttext::loss_name::softmax)
      .value("ova", fasttext::loss_name::ova)
      .export_values();

  py::enum_<fasttext::metric_name>(m, "metric_name")
      .value("f1score", fasttext::metric_name::f1score)
      .value("f1scoreLabel", fasttext::metric_name::f1scoreLabel)
      .value("precisionAtRecall", fasttext::metric_name::precisionAtRecall)
      .value(
          "precisionAtRecallLabel",
          fasttext::metric_name::precisionAtRecallLabel)
      .value("recallAtPrecision", fasttext::metric_name::recallAtPrecision)
      .value(
          "recallAtPrecisionLabel",
          fasttext::metric_name::recallAtPrecisionLabel)
      .export_values();

  m.def(
      "train",
      [](fasttext::FastText& ft, fasttext::Args& a) {
        if (a.hasAutotune()) {
          fasttext::Autotune autotune(std::shared_ptr<fasttext::FastText>(
              &ft, [](fasttext::FastText*) {}));
          autotune.train(a);
        } else {
          ft.train(a);
        }
      },
      py::call_guard<py::gil_scoped_release>());

  py::class_<fasttext::Vector>(m, "Vector", py::buffer_protocol())
      .def(py::init<ssize_t>())
      .def_buffer([](fasttext::Vector& m) -> py::buffer_info {
        return py::buffer_info(
            m.data(),
            sizeof(fasttext::real),
            py::format_descriptor<fasttext::real>::format(),
            1,
            {m.size()},
            {sizeof(fasttext::real)});
      });

  py::class_<fasttext::DenseMatrix>(
      m, "DenseMatrix", py::buffer_protocol(), py::module_local())
      .def(py::init<>())
      .def(py::init<ssize_t, ssize_t>())
      .def_buffer([](fasttext::DenseMatrix& m) -> py::buffer_info {
        return py::buffer_info(
            m.data(),
            sizeof(fasttext::real),
            py::format_descriptor<fasttext::real>::format(),
            2,
            {m.size(0), m.size(1)},
            {sizeof(fasttext::real) * m.size(1),
             sizeof(fasttext::real) * (int64_t)1});
      });

  py::class_<fasttext::Meter>(m, "Meter")
      .def(py::init<bool>())
      .def("scoreVsTrue", &fasttext::Meter::scoreVsTrue)
      .def(
          "precisionRecallCurveLabel",
          (std::vector<std::pair<double, double>>(fasttext::Meter::*)(int32_t)
               const) &
              fasttext::Meter::precisionRecallCurve)
      .def(
          "precisionRecallCurve",
          (std::vector<std::pair<double, double>>(fasttext::Meter::*)() const) &
              fasttext::Meter::precisionRecallCurve)
      .def(
          "precisionAtRecallLabel",
          (double (fasttext::Meter::*)(int32_t, double) const) &
              fasttext::Meter::precisionAtRecall)
      .def(
          "precisionAtRecall",
          (double (fasttext::Meter::*)(double) const) &
              fasttext::Meter::precisionAtRecall)
      .def(
          "recallAtPrecisionLabel",
          (double (fasttext::Meter::*)(int32_t, double) const) &
              fasttext::Meter::recallAtPrecision)
      .def(
          "recallAtPrecision",
          (double (fasttext::Meter::*)(double) const) &
              fasttext::Meter::recallAtPrecision);

  py::class_<fasttext::FastText>(m, "fasttext")
      .def(py::init<>())
      .def("getArgs", &fasttext::FastText::getArgs)
      .def(
          "getInputMatrix",
          [](fasttext::FastText& m) {
            std::shared_ptr<const fasttext::DenseMatrix> mm =
                m.getInputMatrix();
            return mm.get();
          },
          pybind11::return_value_policy::reference)
      .def(
          "getOutputMatrix",
          [](fasttext::FastText& m) {
            std::shared_ptr<const fasttext::DenseMatrix> mm =
                m.getOutputMatrix();
            return mm.get();
          },
          pybind11::return_value_policy::reference)
      .def(
          "setMatrices",
          [](fasttext::FastText& m,
             py::buffer inputMatrixBuffer,
             py::buffer outputMatrixBuffer) {
            py::buffer_info inputMatrixInfo = inputMatrixBuffer.request();
            py::buffer_info outputMatrixInfo = outputMatrixBuffer.request();

            m.setMatrices(
                std::make_shared<fasttext::DenseMatrix>(
                    inputMatrixInfo.shape[0],
                    inputMatrixInfo.shape[1],
                    static_cast<float*>(inputMatrixInfo.ptr)),
                std::make_shared<fasttext::DenseMatrix>(
                    outputMatrixInfo.shape[0],
                    outputMatrixInfo.shape[1],
                    static_cast<float*>(outputMatrixInfo.ptr)));
          })
      .def(
          "loadModel",
          [](fasttext::FastText& m, std::string s) { m.loadModel(s); })
      .def(
          "saveModel",
          [](fasttext::FastText& m, std::string s) { m.saveModel(s); })
      .def(
          "test",
          [](fasttext::FastText& m,
             const std::string& filename,
             int32_t k,
             fasttext::real threshold) {
            std::ifstream ifs(filename);
            if (!ifs.is_open()) {
              throw std::invalid_argument("Test file cannot be opened!");
            }
            fasttext::Meter meter(false);
            m.test(ifs, k, threshold, meter);
            ifs.close();
            return std::tuple<int64_t, double, double>(
                meter.nexamples(), meter.precision(), meter.recall());
          })
      .def(
          "getMeter",
          [](fasttext::FastText& m, const std::string& filename, int32_t k) {
            std::ifstream ifs(filename);
            if (!ifs.is_open()) {
              throw std::invalid_argument("Test file cannot be opened!");
            }
            fasttext::Meter meter(true);
            m.test(ifs, k, 0.0, meter);
            ifs.close();

            return meter;
          })
      .def(
          "getSentenceVector",
          [](fasttext::FastText& m,
             fasttext::Vector& v,
             const std::string text) {
            std::stringstream ioss(text);
            m.getSentenceVector(ioss, v);
          })
      .def(
          "tokenize",
          [](fasttext::FastText& m, const std::string text) {
            std::vector<std::string> text_split;
            std::shared_ptr<const fasttext::Dictionary> d = m.getDictionary();
            std::stringstream ioss(text);
            std::string token;
            while (!ioss.eof()) {
              while (d->readWord(ioss, token)) {
                text_split.push_back(token);
              }
            }
            return text_split;
          })
      .def("getLine", &getLineText)
      .def(
          "multilineGetLine",
          [](fasttext::FastText& m,
             const std::vector<std::string> lines,
             const char* onUnicodeError) {
            std::shared_ptr<const fasttext::Dictionary> d = m.getDictionary();
            std::vector<std::vector<py::str>> all_words;
            std::vector<std::vector<py::str>> all_labels;
            for (const auto& text : lines) {
              auto pair = getLineText(m, text, onUnicodeError);
              all_words.push_back(pair.first);
              all_labels.push_back(pair.second);
            }
            return std::pair<
                std::vector<std::vector<py::str>>,
                std::vector<std::vector<py::str>>>(all_words, all_labels);
          })
      .def(
          "getVocab",
          [](fasttext::FastText& m, const char* onUnicodeError) {
            py::str s;
            std::vector<py::str> vocab_list;
            std::vector<int64_t> vocab_freq;
            std::shared_ptr<const fasttext::Dictionary> d = m.getDictionary();
            vocab_freq = d->getCounts(fasttext::entry_type::word);
            for (int32_t i = 0; i < vocab_freq.size(); i++) {
              vocab_list.push_back(
                  castToPythonString(d->getWord(i), onUnicodeError));
            }
            return std::pair<std::vector<py::str>, std::vector<int64_t>>(
                vocab_list, vocab_freq);
          })
      .def(
          "getLabels",
          [](fasttext::FastText& m, const char* onUnicodeError) {
            std::vector<py::str> labels_list;
            std::vector<int64_t> labels_freq;
            std::shared_ptr<const fasttext::Dictionary> d = m.getDictionary();
            labels_freq = d->getCounts(fasttext::entry_type::label);
            for (int32_t i = 0; i < labels_freq.size(); i++) {
              labels_list.push_back(
                  castToPythonString(d->getLabel(i), onUnicodeError));
            }
            return std::pair<std::vector<py::str>, std::vector<int64_t>>(
                labels_list, labels_freq);
          })
      .def(
          "quantize",
          [](fasttext::FastText& m,
             const std::string input,
             bool qout,
             int32_t cutoff,
             bool retrain,
             int epoch,
             double lr,
             int thread,
             int verbose,
             int32_t dsub,
             bool qnorm) {
            fasttext::Args qa = fasttext::Args();
            qa.input = input;
            qa.qout = qout;
            qa.cutoff = cutoff;
            qa.retrain = retrain;
            qa.epoch = epoch;
            qa.lr = lr;
            qa.thread = thread;
            qa.verbose = verbose;
            qa.dsub = dsub;
            qa.qnorm = qnorm;
            m.quantize(qa);
          })
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
            std::vector<py::array_t<fasttext::real>> allProbabilities;
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

              allProbabilities.emplace_back(
                  probabilities.size(), probabilities.data());
              allLabels.push_back(labels);
            }

            return make_pair(allLabels, allProbabilities);
          })
      .def(
          "testLabel",
          [](fasttext::FastText& m,
             const std::string filename,
             int32_t k,
             fasttext::real threshold) {
            std::ifstream ifs(filename);
            if (!ifs.is_open()) {
              throw std::invalid_argument("Test file cannot be opened!");
            }
            fasttext::Meter meter(false);
            m.test(ifs, k, threshold, meter);
            std::shared_ptr<const fasttext::Dictionary> d = m.getDictionary();
            std::unordered_map<std::string, py::dict> returnedValue;
            for (int32_t i = 0; i < d->nlabels(); i++) {
              returnedValue[d->getLabel(i)] = py::dict(
                  "precision"_a = meter.precision(i),
                  "recall"_a = meter.recall(i),
                  "f1score"_a = meter.f1Score(i));
            }

            return returnedValue;
          })
      .def(
          "getWordId",
          [](fasttext::FastText& m, const std::string& word) {
            return m.getWordId(word);
          })
      .def(
          "getSubwordId",
          [](fasttext::FastText& m, const std::string word) {
            return m.getSubwordId(word);
          })
      .def(
          "getLabelId",
          [](fasttext::FastText& m, const std::string& label) {
            return m.getLabelId(label);
          })
      .def(
          "getInputVector",
          [](fasttext::FastText& m, fasttext::Vector& vec, int32_t ind) {
            m.getInputVector(vec, ind);
          })
      .def(
          "getWordVector",
          [](fasttext::FastText& m,
             fasttext::Vector& vec,
             const std::string word) { m.getWordVector(vec, word); })
      .def(
          "getNN",
          [](fasttext::FastText& m,
             const std::string& word,
             int32_t k,
             const char* onUnicodeError) {
            return castToPythonString(m.getNN(word, k), onUnicodeError);
          })
      .def(
          "getAnalogies",
          [](fasttext::FastText& m,
             const std::string& wordA,
             const std::string& wordB,
             const std::string& wordC,
             int32_t k,
             const char* onUnicodeError) {
            return castToPythonString(
                m.getAnalogies(k, wordA, wordB, wordC), onUnicodeError);
          })
      .def(
          "getSubwords",
          [](fasttext::FastText& m,
             const std::string word,
             const char* onUnicodeError) {
            std::vector<std::string> subwords;
            std::vector<int32_t> ngrams;
            std::shared_ptr<const fasttext::Dictionary> d = m.getDictionary();
            d->getSubwords(word, ngrams, subwords);
            std::vector<py::str> transformedSubwords;

            for (const auto& subword : subwords) {
              transformedSubwords.push_back(
                  castToPythonString(subword, onUnicodeError));
            }

            return std::pair<std::vector<py::str>, std::vector<int32_t>>(
                transformedSubwords, ngrams);
          })
      .def("isQuant", [](fasttext::FastText& m) { return m.isQuant(); });
}
