
#ifndef DUNEDAQ_DTPEMULATOR_TPGENERATOR_HPP_
#define DUNEDAQ_DTPEMULATOR_TPGENERATOR_HPP_

#include <string>
#include <memory>
#include <fstream>
#include <vector>
#include <sstream>
#include <iterator>
#include <bitset>
#include <algorithm>
#include <functional>
#include <numeric>

namespace dunedaq
{
  namespace DTPEMULATOR
  {

    class TPGenerator
    {
    public:
      TPGenerator(const unsigned int initial_pedestal, const std::string fir_data, const unsigned int fir_shift, const unsigned int threshold);
      ~TPGenerator();

      std::vector<std::vector<int>> pedestal_subtraction(std::vector<int> adcs, int limit = 10);
      std::vector<int> fir_filter(std::vector<int> adcs);
      std::vector<std::vector<int>> hit_finder(std::vector<int> adcs, int tov_min = 4);

    private:
      unsigned int m_initial_pedestal;
      std::vector<int> m_fir_coeffs;
      unsigned int m_fir_shift;
      unsigned int m_threshold;
    };

  }
}

#endif /* DUNEDAQ_DTPEMULATOR_TPGENERATOR_HPP_ */