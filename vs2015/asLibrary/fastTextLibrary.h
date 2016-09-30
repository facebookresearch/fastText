#pragma once

#include <fenv.h>
#include <time.h>
#include <math.h>

#include <map>
#include <iostream>
#include <iomanip>
#include <thread>
#include <string>
#include <vector>
#include <atomic>
#include <algorithm>
#include <sstream>
#include <istream>
#include <iostream>

#include "matrix.h"
#include "vector.h"
#include "dictionary.h"
#include "model.h"
#include "utils.h"
#include "real.h"
#include "args.h"

class fastTextModelData
{
public:
    std::shared_ptr<Args>       args;
    std::shared_ptr<Dictionary> dict;
    std::shared_ptr<Matrix>     input;
    std::shared_ptr<Matrix>     output;
    std::shared_ptr<Model>      model;
    std::string                 path;
    bool modelInitialized;
    std::string                 lastPrediction;
};
