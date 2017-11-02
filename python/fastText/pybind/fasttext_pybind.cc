/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include <args.h>
#include <fasttext.h>
#include <matrix.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <real.h>
#include <vector.h>
#include <iterator>
#include <sstream>

namespace py = pybind11;

PYBIND11_MODULE(fasttext_pybind, m) {
  py::class_<fasttext::Args>(m, "args")
      .def(py::init<>())
      .def_readwrite("input", &fasttext::Args::input)
      .def_readwrite("test", &fasttext::Args::test)
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

      .def_readwrite("qout", &fasttext::Args::qout)
      .def_readwrite("retrain", &fasttext::Args::retrain)
      .def_readwrite("qnorm", &fasttext::Args::qnorm)
      .def_readwrite("cutoff", &fasttext::Args::cutoff)
      .def_readwrite("dsub", &fasttext::Args::dsub);

  py::enum_<fasttext::model_name>(m, "model_name")
      .value("cbow", fasttext::model_name::cbow)
      .value("skipgram", fasttext::model_name::sg)
      .value("supervised", fasttext::model_name::sup)
      .export_values();

  py::enum_<fasttext::loss_name>(m, "loss_name")
      .value("hs", fasttext::loss_name::hs)
      .value("ns", fasttext::loss_name::ns)
      .value("softmax", fasttext::loss_name::softmax)
      .export_values();

  m.def("train", [](fasttext::FastText& ft, fasttext::Args& a) {
    std::shared_ptr<fasttext::Args> sa = std::make_shared<fasttext::Args>(a);
    ft.train(sa);
  });

  py::class_<fasttext::Vector>(m, "Vector", py::buffer_protocol())
      .def(py::init<ssize_t>())
      .def_buffer([](fasttext::Vector& m) -> py::buffer_info {
        return py::buffer_info(
            m.data_,
            sizeof(fasttext::real),
            py::format_descriptor<fasttext::real>::format(),
            1,
            {m.m_},
            {sizeof(fasttext::real)});
      });

  py::class_<fasttext::Matrix>(
      m, "Matrix", py::buffer_protocol(), py::module_local())
      .def(py::init<>())
      .def(py::init<ssize_t, ssize_t>())
      .def_buffer([](fasttext::Matrix& m) -> py::buffer_info {
        return py::buffer_info(
            m.data_,
            sizeof(fasttext::real),
            py::format_descriptor<fasttext::real>::format(),
            2,
            {m.m_, m.n_},
            {sizeof(fasttext::real) * m.n_, sizeof(fasttext::real) * (int64_t)1});
      });

  py::class_<fasttext::FastText>(m, "fasttext")
      .def(py::init<>())
      .def("getArgs", &fasttext::FastText::getArgs)
      .def(
          "getInputMatrix",
          [](fasttext::FastText& m) {
            std::shared_ptr<const fasttext::Matrix> mm = m.getInputMatrix();
            return *mm.get();
          })
      .def(
          "getOutputMatrix",
          [](fasttext::FastText& m) {
            std::shared_ptr<const fasttext::Matrix> mm = m.getOutputMatrix();
            return *mm.get();
          })
      .def(
          "loadModel",
          [](fasttext::FastText& m, std::string s) { m.loadModel(s); })
      .def(
          "saveModel",
          [](fasttext::FastText& m, std::string s) { m.saveModel(s); })
      .def(
          "getSentenceVector",
          [](fasttext::FastText& m,
             fasttext::Vector& v,
             const std::string text) {
            std::stringstream ioss;
            copy(text.begin(), text.end(), std::ostream_iterator<char>(ioss));
            m.getSentenceVector(ioss, v);
          })
      .def(
          "tokenize",
          [](fasttext::FastText& m, const std::string text) {
            std::vector<std::string> text_split;
            std::shared_ptr<const fasttext::Dictionary> d = m.getDictionary();
            std::stringstream ioss;
            copy(text.begin(), text.end(), std::ostream_iterator<char>(ioss));
            std::string token;
            while (!ioss.eof()) {
              while (d->readWord(ioss, token)) {
                text_split.push_back(token);
              }
            }
            return text_split;
          })
      .def(
          "getVocab",
          [](fasttext::FastText& m) {
            std::vector<std::string> vocab_list;
            std::vector<int64_t> vocab_freq;
            std::shared_ptr<const fasttext::Dictionary> d = m.getDictionary();
            vocab_freq = d->getCounts(fasttext::entry_type::word);
            vocab_list.clear();
            for (int32_t i = 0; i < vocab_freq.size(); i++) {
              vocab_list.push_back(d->getWord(i));
            }
            return std::pair<std::vector<std::string>, std::vector<int64_t>>(
                vocab_list, vocab_freq);
          })
      .def(
          "getLabels",
          [](fasttext::FastText& m) {
            std::vector<std::string> labels_list;
            std::vector<int64_t> labels_freq;
            std::shared_ptr<const fasttext::Dictionary> d = m.getDictionary();
            labels_freq = d->getCounts(fasttext::entry_type::label);
            labels_list.clear();
            for (int32_t i = 0; i < labels_freq.size(); i++) {
              labels_list.push_back(d->getLabel(i));
            }
            return std::pair<std::vector<std::string>, std::vector<int64_t>>(
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
            std::shared_ptr<fasttext::Args> qa =
                std::make_shared<fasttext::Args>();
            qa->input = input;
            qa->qout = qout;
            qa->cutoff = cutoff;
            qa->retrain = retrain;
            qa->epoch = epoch;
            qa->lr = lr;
            qa->thread = thread;
            qa->verbose = verbose;
            qa->dsub = dsub;
            qa->qnorm = qnorm;
            m.quantize(qa);
          })
      .def(
          "predict",
          // NOTE: text needs to end in a newline
          // to exactly mimic the behavior of the cli
          [](fasttext::FastText& m, const std::string text, int32_t k) {
            std::vector<std::pair<fasttext::real, std::string>> predictions;
            std::stringstream ioss;
            copy(text.begin(), text.end(), std::ostream_iterator<char>(ioss));
            m.predict(ioss, k, predictions);
            return predictions;
          })
      .def(
          "isQuant",
          [](fasttext::FastText& m) {
            return m.isQuant();
          })
      .def(
          "getWordId",
          [](fasttext::FastText& m, const std::string word) {
            return m.getWordId(word);
          })
      .def(
          "getSubwordId",
          [](fasttext::FastText& m, const std::string word) {
            return m.getSubwordId(word);
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
          "getSubwords",
          [](fasttext::FastText& m, const std::string word) {
            std::vector<std::string> subwords;
            std::vector<int32_t> ngrams;
            std::shared_ptr<const fasttext::Dictionary> d = m.getDictionary();
            d->getSubwords(word, ngrams, subwords);
            return std::pair<std::vector<std::string>, std::vector<int32_t>>(
                subwords, ngrams);
          })
      .def("isQuant", [](fasttext::FastText& m) { return m.isQuant(); });
}
