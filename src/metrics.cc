#include "metrics.h"

#include <algorithm>
#include <cmath>
#include <iomanip>

namespace fasttext {

/* ----- MetricsAccumulator ----- */

void MetricsAccumulator::log(
    const std::vector<int32_t>& labels,
    const std::vector<std::pair<real, int32_t>>& predictions) {
  metrics_.numExamples++;
  metrics_.numLabels += labels.size();
  metrics_.numPredictions += predictions.size();
  metrics_.numTruePositives += std::count_if(
      predictions.begin(),
      predictions.end(),
      [&](std::pair<real, int32_t> prediction) {
        return std::find(labels.begin(), labels.end(), prediction.second) !=
            labels.end();
      });
}

/* ----- LabelMetricsAccumulator ----- */

void LabelMetricsAccumulator::log(
    const std::vector<int32_t>& labels,
    const std::vector<std::pair<real, int32_t>>& predictions) {
  MetricsAccumulator::log(labels, predictions);

  for (const auto& prediction : predictions) {
    labelMetrics_[prediction.second].numPredictions++;

    if (std::find(labels.begin(), labels.end(), prediction.second) !=
        labels.end())
      labelMetrics_[prediction.second].numTruePositives++;
    else
      labelMetrics_[prediction.second].numExamples++;
  }

  for (const auto& label : labels) {
    labelMetrics_[label].numLabels++;
    labelMetrics_[label].numExamples++;
  }
}

void LabelMetricsAccumulator::write(
    std::ostream& out,
    std::shared_ptr<const Dictionary> dict) const {
  out << std::fixed;
  out << std::setprecision(6);

  auto writeMetric = [&](const std::string& name, double value) {
    out << name << " : ";
    if (std::isfinite(value))
      out << value;
    else
      out << "--------";
    out << "  ";
  };

  for (int32_t i = 0; i < dict->nlabels(); i++) {
    auto it = labelMetrics_.find(i);
    if (it != labelMetrics_.end()) {
      const auto& metrics = it->second;
      writeMetric("F1-Score", metrics.f1Score());
      writeMetric("Precision", metrics.precision());
      writeMetric("Recall", metrics.recall());
      out << " " << dict->getLabel(i) << std::endl;
    } else {
      writeMetric("F1-Score", NAN);
      writeMetric("Precision", NAN);
      writeMetric("Recall", NAN);
      out << " " << dict->getLabel(i) << std::endl;
    }
  }
}

} // namespace fasttext
