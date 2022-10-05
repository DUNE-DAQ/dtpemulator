#include "dtpemulator/TPGenerator.hpp"

#include <iostream>

namespace dunedaq
{
  namespace dtpemulator
  {

    TPGenerator::TPGenerator(const std::string fir_data, const unsigned int fir_shift, const unsigned int threshold)
    {

      std::ifstream is(fir_data);
      double tap;
      while ( is >> tap ) {
        //std::cout << tap << std::endl;
        m_fir_coeffs.push_back(tap);
      }
      //std::cout << "Number of taps " << m_fir_coeffs.size() << std::endl;

      m_fir_shift = fir_shift;
      //std::cout << "FIR shift " << m_fir_shift << std::endl;

      m_threshold = threshold;
      //std::cout << "Threshold " << m_threshold << std::endl;

    }

    TPGenerator::~TPGenerator() {
      
    }

    std::vector<std::vector<int>>
    TPGenerator::pedestal_subtraction(std::vector<int> adcs, int ini_median, int ini_accum, int limit) {

      int median = ini_median;
      int accumulator = ini_accum;
      std::vector<int> pedsub;
      std::vector<int> pedval;
      std::vector<int> accum;

      for (long unsigned int iadc = 0; iadc < adcs.size(); iadc++) {

        if (adcs[iadc] > median) {
          accumulator += 1;
        } else if (adcs[iadc] < median) {
          accumulator -= 1;
        } else {
          accumulator = accumulator;
        }

        if (accumulator == limit) {
          accumulator = 0;
          median += 1;

        } else if (accumulator == -limit) {
          accumulator = 0;
          median -= 1;
        }

        pedsub.push_back(adcs[iadc] - median);

        if (iadc < adcs.size()){
          pedval.push_back(median);
          accum.push_back(accumulator);
        }

      }

      std::vector<std::vector<int>> pedestal = {pedsub, pedval, accum};

      return pedestal;
    }

    std::vector<int>
    TPGenerator::fir_filter(std::vector<int> adcs) {

      std::vector<int> fir;

      for (long unsigned int iadc = 0; iadc < adcs.size(); iadc++){

        int tmp = 0;

        for (long unsigned int itap = 0; itap < m_fir_coeffs.size(); itap++){

          int index = (iadc >= itap) ? iadc-itap : false;
          //std::cout << "iadc " << iadc << "itap " << itap << "indx " << index << std::endl;

          if (index){
            tmp += adcs[index]*m_fir_coeffs[itap];
          }
        }

        fir.push_back(tmp >> m_fir_shift);
      }

      return fir;
    }

    std::vector<std::vector<int>>
    TPGenerator::hit_finder(std::vector<int> adcs, int tov_min){

      std::vector<std::vector<int>> hits;

      // index of ADCs above threshold
      int threshold = (int)m_threshold;

      std::vector<int> igt;
      std::vector<int>::iterator it_adcs = adcs.begin();
      while ((it_adcs = std::find_if(it_adcs, adcs.end(), [&threshold](int x){return x >= threshold; })) != adcs.end())
      {
        igt.push_back(std::distance(adcs.begin(), it_adcs));
        it_adcs++;
      }

      if(igt.size() < tov_min){return std::vector<std::vector<int>>();}

      std::vector<int> igt_diff;
      std::adjacent_difference (igt.begin(), igt.end(), std::back_inserter(igt_diff));
      igt_diff.erase(igt_diff.begin());

      // find start and end of hits
      std::vector<int> istart;
      std::vector<int> iend;
      istart.push_back(0);
      std::vector<int>::iterator it_igt = igt_diff.begin();
      while ((it_igt = std::find_if(it_igt, igt_diff.end(), [ ](int x){return x != 1; })) != igt_diff.end())
      {
        istart.push_back(std::distance(igt_diff.begin(), it_igt)+1);
        iend.push_back(std::distance(igt_diff.begin(), it_igt));
        it_igt++;
      }
      iend.push_back(igt.size()-1);

      std::vector<int> start;
      std::vector<int> end;
      std::vector<int> hitcontinue;
      for(long unsigned int i = 0; i < istart.size(); i++){
        start.push_back(igt[istart[i]]);
        end.push_back(igt[iend[i]]);
        if(end[i] == 63){
          hitcontinue.push_back(1);
        }else{
          hitcontinue.push_back(0);
        }
      }

      // find hit sums
      std::vector<int> sums;
      for(long unsigned int j = 0; j < start.size(); j++){
        std::vector<int>::iterator it_start = adcs.begin()+start[j];
        std::vector<int>::iterator it_end = adcs.begin()+end[j];
        int sum = std::accumulate(it_start, it_end, 0);
        sums.push_back(sum);
      }

      // find peak adcs and times
      std::vector<int> peak_adcs;
      std::vector<int> peak_times;
      for(long unsigned int j = 0; j < start.size(); j++){
        std::vector<int>::iterator it_start = adcs.begin()+start[j];
        std::vector<int>::iterator it_end = adcs.begin()+end[j];
        std::vector<int>::iterator max = std::max_element(it_start, it_end);
        int peak = *max;
        int time = distance(adcs.begin(), max);
        peak_adcs.push_back(peak);
        peak_times.push_back(time);
      }

      // check output hits fullfil the tov_min condition
      std::vector<std::vector<int>> out;
      for(long unsigned int k = 0; k < start.size(); k++){
        if (end[k]-start[k] > tov_min){
          std::vector<int> aux = {start[k], end[k], peak_times[k], peak_adcs[k], sums[k], hitcontinue[k]};
          out.push_back(aux);
          //std::cout << "# hit " << k << ", start time " << start[k] << ", end time " << end[k] << ", peak time " << peak_times[k] << ", peak adc " << peak_adcs[k] << ", sum adc " << sums[k] << ", hit continue " << hitcontinue[k] << std::endl;
        }
      }

      return out;

    }

  } // namespace dtpemulator
} // namespace dunedaq
