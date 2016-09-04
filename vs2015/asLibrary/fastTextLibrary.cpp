/**
* Copyright (c) 2016-present, Facebook, Inc.
* Copyright (c) 2016-present, Rafael Fernandes de Oliveira - rafael@rafael.aero
* All rights reserved.
*
* This source code is licensed under the BSD-style license found in the
* LICENSE file in the root directory of this source tree. An additional grant
* of patent rights can be found in the PATENTS file in the same directory.
*/

#include "fastTextLibrary.h"

Args                        args;


std::vector<std::shared_ptr<fastTextModelData>> models;

extern "C" __declspec(dllexport) void __cdecl initialize()
{
    utils::initTables();
}


extern "C" __declspec(dllexport) void __cdecl dispose()
{
    utils::freeTables();
}

extern "C" __declspec(dllexport) int __cdecl loadModel(const char *filename)
{
    auto& modelData = std::make_shared<fastTextModelData>();

    modelData->args   = std::make_shared<Args>();
    modelData->dict   = std::make_shared<Dictionary>();
    modelData->input  = std::make_shared<Matrix>();
    modelData->output = std::make_shared<Matrix>();


    std::ifstream ifs(filename, std::ifstream::binary);

    if (!ifs.is_open()) {
        std::cerr << "Model file cannot be opened for loading!" << std::endl;
        return false;
    }

    modelData->args->load(ifs);
    modelData->dict->load(ifs);
    modelData->input->load(ifs);
    modelData->output->load(ifs);

    ifs.close();
    modelData->modelInitialized = false;

    modelData->path = std::string(filename);

    models.push_back(modelData);

    return models.size() - 1;
}


std::shared_ptr<fastTextModelData> getModel(int model_index)
{
    auto& model = models[model_index];
    args = *(model->args.get());
    return model;
}


void initializeModelForPredict(int model_index)
{
    auto& model = getModel(model_index);
    model->model = std::make_shared<Model>(*(model->input.get()), *(model->output.get()), args.dim, args.lr, 1);
    model->model->setTargetCounts(model->dict->getCounts(entry_type::label));
    model->modelInitialized = true;
}


extern "C" __declspec(dllexport) void __cdecl predict(int model_index, const char *text, const int32_t k)
{
    auto& model = getModel(model_index);

    if (!model->modelInitialized) { initializeModelForPredict(model_index); }

    std::string inputstr(text);
    std::istringstream ifs(inputstr);
    std::vector<int32_t> line, labels;
    std::ostringstream oss;

    while (ifs.peek() != EOF) 
    {
        model->dict->getLine(ifs, line, labels, model->model->rng);
        model->dict->addNgrams(line, args.wordNgrams);

        if (line.empty()) 
        {
            oss << "n/a" << std::endl;
            continue;
        }

        std::vector<std::pair<real, int32_t>> predictions;
        model->model->predict(line, k, predictions);

        for (auto it = predictions.cbegin(); it != predictions.cend(); it++) 
        {
            if (it != predictions.cbegin()) { oss << ';'; }

            oss << model->dict->getLabel(it->second) << "[" << std::to_string(it->first) << "]";
        }
        oss << std::endl;
    }

    model->lastPrediction = oss.str();
    
    //output.copy(out,output.size());
}

extern "C" __declspec(dllexport) int __cdecl  getPredictionBufferSize(int model_index)
{
    auto& model = getModel(model_index);
    return model->lastPrediction.size();
}

extern "C" __declspec(dllexport) void __cdecl getPrediction(int model_index, char* out)
{
    auto& model = getModel(model_index);
    std::strcpy(out, model->lastPrediction.c_str());
}

extern "C" __declspec(dllexport) int __cdecl getVectorSize(int model_index)
{
    auto& model = getModel(model_index);
    return args.dim;
}

extern "C" __declspec(dllexport) void __cdecl getVector(int model_index, const char *word, double* out)
{
    auto& model = getModel(model_index);

    Vector vec(args.dim);
    std::string word_str(word);

    const std::vector<int32_t> ngrams = model->dict->getNgrams(word_str);
    vec.zero();
    for (auto it = ngrams.begin(); it != ngrams.end(); ++it) 
    {
        vec.addRow(*(model->input.get()), *it);
    }
    if (ngrams.size() > 0) {
        vec.mul(1.0 / ngrams.size());
    }

    for (int i = 0; i < args.dim; i++)
    {
        out[i] = vec[i];
    }
}



#ifdef OLD_CODE
Args args;

namespace info {
    clock_t start;
    std::atomic<int64_t> allWords(0);
    std::atomic<int64_t> allN(0);
    double allLoss(0.0);
}

void getVector(Dictionary& dict, Matrix& input, Vector& vec, std::string word) {
    const std::vector<int32_t>& ngrams = dict.getNgrams(word);
    vec.zero();
    for (auto it = ngrams.begin(); it != ngrams.end(); ++it) {
        vec.addRow(input, *it);
    }
    if (ngrams.size() > 0) {
        vec.mul(1.0 / ngrams.size());
    }
}

void saveVectors(Dictionary& dict, Matrix& input, Matrix& output) {
    std::ofstream ofs(args.output + ".vec");
    if (!ofs.is_open()) {
        std::cout << "Error opening file for saving vectors." << std::endl;
        exit(EXIT_FAILURE);
    }
    ofs << dict.nwords() << " " << args.dim << std::endl;
    Vector vec(args.dim);
    for (int32_t i = 0; i < dict.nwords(); i++) {
        std::string word = dict.getWord(i);
        getVector(dict, input, vec, word);
        ofs << word << " " << vec << std::endl;
    }
    ofs.close();
}

void printVectors(Dictionary& dict, Matrix& input) {
    std::string word;
    Vector vec(args.dim);
    while (std::cin >> word) {
        getVector(dict, input, vec, word);
        std::cout << word << " " << vec << std::endl;
    }
}

void saveModel(Dictionary& dict, Matrix& input, Matrix& output) {
    std::ofstream ofs(args.output + ".bin", std::ofstream::binary);
    if (!ofs.is_open()) {
        std::cerr << "Model file cannot be opened for saving!" << std::endl;
        exit(EXIT_FAILURE);
    }
    args.save(ofs);
    dict.save(ofs);
    input.save(ofs);
    output.save(ofs);
    ofs.close();
}

void loadModel(std::string filename, Dictionary& dict,
               Matrix& input, Matrix& output) {
    std::ifstream ifs(filename, std::ifstream::binary);
    if (!ifs.is_open()) {
        std::cerr << "Model file cannot be opened for loading!" << std::endl;
        exit(EXIT_FAILURE);
    }
    args.load(ifs);
    dict.load(ifs);
    input.load(ifs);
    output.load(ifs);
    ifs.close();
}

void printInfo(Model& model, real progress) {
    real loss = info::allLoss / info::allN;
    real t = real(clock() - info::start) / CLOCKS_PER_SEC;
    real wst = real(info::allWords) / t;
    int eta = int(t / progress * (1 - progress) / args.thread);
    int etah = eta / 3600;
    int etam = (eta - etah * 3600) / 60;
    std::cout << std::fixed;
    std::cout << "\rProgress: " << std::setprecision(1) << 100 * progress << "%";
    std::cout << "  words/sec/thread: " << std::setprecision(0) << wst;
    std::cout << "  lr: " << std::setprecision(6) << model.getLearningRate();
    std::cout << "  loss: " << std::setprecision(6) << loss;
    std::cout << "  eta: " << etah << "h" << etam << "m ";
    std::cout << std::flush;
}

void supervised(Model& model,
                const std::vector<int32_t>& line,
                const std::vector<int32_t>& labels,
                double& loss, int32_t& nexamples) {
    if (labels.size() == 0 || line.size() == 0) return;
    std::uniform_int_distribution<> uniform(0, labels.size() - 1);
    int32_t i = uniform(model.rng);
    loss += model.update(line, labels[i]);
    nexamples++;
}

void cbow(Dictionary& dict, Model& model,
          const std::vector<int32_t>& line,
          double& loss, int32_t& nexamples) {
    std::vector<int32_t> bow;
    std::uniform_int_distribution<> uniform(1, args.ws);
    for (int32_t w = 0; w < line.size(); w++) {
        int32_t boundary = uniform(model.rng);
        bow.clear();
        for (int32_t c = -boundary; c <= boundary; c++) {
            if (c != 0 && w + c >= 0 && w + c < line.size()) {
                const std::vector<int32_t>& ngrams = dict.getNgrams(line[w + c]);
                bow.insert(bow.end(), ngrams.cbegin(), ngrams.cend());
            }
        }
        loss += model.update(bow, line[w]);
        nexamples++;
    }
}

void skipgram(Dictionary& dict, Model& model,
              const std::vector<int32_t>& line,
              double& loss, int32_t& nexamples) {
    std::uniform_int_distribution<> uniform(1, args.ws);
    for (int32_t w = 0; w < line.size(); w++) {
        int32_t boundary = uniform(model.rng);
        const std::vector<int32_t>& ngrams = dict.getNgrams(line[w]);
        for (int32_t c = -boundary; c <= boundary; c++) {
            if (c != 0 && w + c >= 0 && w + c < line.size()) {
                loss += model.update(ngrams, line[w + c]);
                nexamples++;
            }
        }
    }
}

void test(Dictionary& dict, Model& model, std::string filename, int32_t k) {
    int32_t nexamples = 0, nlabels = 0;
    double precision = 0.0;
    std::vector<int32_t> line, labels;
    std::ifstream ifs(filename);
    if (!ifs.is_open()) {
        std::cerr << "Test file cannot be opened!" << std::endl;
        exit(EXIT_FAILURE);
    }
    while (ifs.peek() != EOF) {
        dict.getLine(ifs, line, labels, model.rng);
        dict.addNgrams(line, args.wordNgrams);
        if (labels.size() > 0 && line.size() > 0) {
            std::vector<std::pair<real, int32_t>> predictions;
            model.predict(line, k, predictions);
            for (auto it = predictions.cbegin(); it != predictions.cend(); it++) {
                if (std::find(labels.begin(), labels.end(), it->second) != labels.end()) {
                    precision += 1.0;
                }
            }
            nexamples++;
            nlabels += labels.size();
        }
    }
    ifs.close();
    std::cout << std::setprecision(3);
    std::cout << "P@" << k << ": " << precision / (k * nexamples) << std::endl;
    std::cout << "R@" << k << ": " << precision / nlabels << std::endl;
    std::cout << "Number of examples: " << nexamples << std::endl;
}

void predict(Dictionary& dict, Model& model, std::string filename, int32_t k) {
    std::vector<int32_t> line, labels;
    std::ifstream ifs(filename);
    if (!ifs.is_open()) {
        std::cerr << "Test file cannot be opened!" << std::endl;
        exit(EXIT_FAILURE);
    }
    while (ifs.peek() != EOF) {
        dict.getLine(ifs, line, labels, model.rng);
        dict.addNgrams(line, args.wordNgrams);
        if (line.empty()) {
            std::cout << "n/a" << std::endl;
            continue;
        }
        std::vector<std::pair<real, int32_t>> predictions;
        model.predict(line, k, predictions);
        for (auto it = predictions.cbegin(); it != predictions.cend(); it++) {
            if (it != predictions.cbegin()) {
                std::cout << ' ';
            }
            std::cout << dict.getLabel(it->second);
        }
        std::cout << std::endl;
    }
    ifs.close();
}

void trainThread(Dictionary& dict, Matrix& input, Matrix& output,
                 int32_t threadId) {
    std::ifstream ifs(args.input);
    utils::seek(ifs, threadId * utils::size(ifs) / args.thread);

    Model model(input, output, args.dim, args.lr, threadId);
    if (args.model == model_name::sup) {
        model.setTargetCounts(dict.getCounts(entry_type::label));
    } else {
        model.setTargetCounts(dict.getCounts(entry_type::word));
    }

    real progress;
    const int64_t ntokens = dict.ntokens();
    int64_t tokenCount = 0, printCount = 0, deltaCount = 0;
    double loss = 0.0;
    int32_t nexamples = 0;
    std::vector<int32_t> line, labels;
    while (info::allWords < args.epoch * ntokens) {
        deltaCount = dict.getLine(ifs, line, labels, model.rng);
        tokenCount += deltaCount;
        printCount += deltaCount;
        if (args.model == model_name::sup) {
            dict.addNgrams(line, args.wordNgrams);
            supervised(model, line, labels, loss, nexamples);
        } else if (args.model == model_name::cbow) {
            cbow(dict, model, line, loss, nexamples);
        } else if (args.model == model_name::sg) {
            skipgram(dict, model, line, loss, nexamples);
        }
        if (tokenCount > args.lrUpdateRate) {
            info::allWords += tokenCount;
            info::allLoss += loss;
            info::allN += nexamples;
            tokenCount = 0;
            loss = 0.0;
            nexamples = 0;
            progress = real(info::allWords) / (args.epoch * ntokens);
            model.setLearningRate(args.lr * (1.0 - progress));
            if (threadId == 0) {
                printInfo(model, progress);
            }
        }
    }
    if (threadId == 0) {
        printInfo(model, 1.0);
        std::cout << std::endl;
    }
    ifs.close();
}

void printUsage() {
    std::cout
        << "usage: fasttext <command> <args>\n\n"
        << "The commands supported by fasttext are:\n\n"
        << "  supervised       train a supervised classifier\n"
        << "  test             evaluate a supervised classifier\n"
        << "  predict          predict most likely label\n"
        << "  skipgram         train a skipgram model\n"
        << "  cbow             train a cbow model\n"
        << "  print-vectors    print vectors given a trained model\n"
        << std::endl;
}

void printTestUsage() {
    std::cout
        << "usage: fasttext test <model> <test-data> [<k>]\n\n"
        << "  <model>      model filename\n"
        << "  <test-data>  test data filename\n"
        << "  <k>          (optional; 1 by default) predict top k labels\n"
        << std::endl;
}

void printPredictUsage() {
    std::cout
        << "usage: fasttext predict <model> <test-data> [<k>]\n\n"
        << "  <model>      model filename\n"
        << "  <test-data>  test data filename\n"
        << "  <k>          (optional; 1 by default) predict top k labels\n"
        << std::endl;
}

void printPrintVectorsUsage() {
    std::cout
        << "usage: fasttext print-vectors <model>\n\n"
        << "  <model>      model filename\n"
        << std::endl;
}

void test(int argc, char** argv) {
    int32_t k;
    if (argc == 4) {
        k = 1;
    } else if (argc == 5) {
        k = atoi(argv[4]);
    } else {
        printTestUsage();
        exit(EXIT_FAILURE);
    }
    Dictionary dict;
    Matrix input, output;
    loadModel(std::string(argv[2]), dict, input, output);
    Model model(input, output, args.dim, args.lr, 1);
    model.setTargetCounts(dict.getCounts(entry_type::label));
    test(dict, model, std::string(argv[3]), k);
    exit(0);
}

void predict(int argc, char** argv) {
    int32_t k;
    if (argc == 4) {
        k = 1;
    } else if (argc == 5) {
        k = atoi(argv[4]);
    } else {
        printPredictUsage();
        exit(EXIT_FAILURE);
    }
    Dictionary dict;
    Matrix input, output;
    loadModel(std::string(argv[2]), dict, input, output);
    Model model(input, output, args.dim, args.lr, 1);
    model.setTargetCounts(dict.getCounts(entry_type::label));
    predict(dict, model, std::string(argv[3]), k);
    exit(0);
}

void printVectors(int argc, char** argv) {
    if (argc != 3) {
        printPrintVectorsUsage();
        exit(EXIT_FAILURE);
    }
    Dictionary dict;
    Matrix input, output;
    loadModel(std::string(argv[2]), dict, input, output);
    printVectors(dict, input);
    exit(0);
}

void train(int argc, char** argv) {
    args.parseArgs(argc, argv);

    Dictionary dict;
    std::ifstream ifs(args.input);
    if (!ifs.is_open()) {
        std::cerr << "Input file cannot be opened!" << std::endl;
        exit(EXIT_FAILURE);
    }
    dict.readFromFile(ifs);
    ifs.close();

    Matrix input(dict.nwords() + args.bucket, args.dim);
    Matrix output;
    if (args.model == model_name::sup) {
        output = Matrix(dict.nlabels(), args.dim);
    } else {
        output = Matrix(dict.nwords(), args.dim);
    }
    input.uniform(1.0 / args.dim);
    output.zero();

    info::start = clock();
    time_t t0 = time(nullptr);
    std::vector<std::thread> threads;
    for (int32_t i = 0; i < args.thread; i++) {
        threads.push_back(std::thread(&trainThread, std::ref(dict),
                                      std::ref(input), std::ref(output), i));
    }
    for (auto it = threads.begin(); it != threads.end(); ++it) {
        it->join();
    }
    double trainTime = difftime(time(nullptr), t0);
    std::cout << "Train time: " << trainTime << " sec" << std::endl;

    saveModel(dict, input, output);
    if (args.model != model_name::sup) {
        saveVectors(dict, input, output);
    }
}

int main(int argc, char** argv) {
    utils::initTables();
    if (argc < 2) {
        printUsage();
        exit(EXIT_FAILURE);
    }
    std::string command(argv[1]);
    if (command == "skipgram" || command == "cbow" || command == "supervised") {
        train(argc, argv);
    } else if (command == "test") {
        test(argc, argv);
    } else if (command == "print-vectors") {
        printVectors(argc, argv);
    } else if (command == "predict") {
        predict(argc, argv);
    } else {
        printUsage();
        exit(EXIT_FAILURE);
    }
    utils::freeTables();
    return 0;
}

#endif