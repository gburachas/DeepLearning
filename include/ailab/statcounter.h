#ifndef STATCOUNTER_H
#define STATCOUNTER_H

#include <cmath>

namespace ailab{

class StatCounter{

 protected:

  double oldMean;
  double newMean;
  double oldSum;
  double newSum;
  double newest;
  double stdevSum;
  double _min;
  double _max;
  size_t count;

 public:

    StatCounter() :
     oldMean(0), newMean(0), oldSum(0), newSum(0), count(0)
     , newest(0)
 , _min(std::numeric_limits<double>::infinity())
 , _max(-std::numeric_limits<double>::infinity()){}

    void push(double value){

      this->newest = value;
      this->_min = std::min(value, this->_min);
      this->_max = std::max(value, this->_max);

      this->count++;
      if(this->count==1){
        this->oldMean = this->newMean = value;
        this->oldSum=0;
      }else{
        this->newMean = this->oldMean + (value - this->oldMean)/this->count;
        this->newSum = this->oldSum + (value - this->oldMean) * (value - this->newMean);
        this->oldMean = this->newMean;
        this->oldSum = this->newSum;
      }

      this->stdevSum += pow(value - this->newMean, 2);
    }

    double front(){ return this->newest; }

    /**
   * @brief stdev
   * @return An estimate of the standard deviation, assuming a gaussian distribution
   */
  double stdev(){
    return (this->count > 1)? std::sqrt( this->stdevSum / (this->count - 1.5) ) : 0.0;
  }

  double mean(){
    return this->newMean;
  }

  double min(){ return this->_min; }
  double max(){ return this->_max; }

  size_t nObs(){ return this->count; }

  void reset(){
    this->oldMean = 0;
    this->newMean = 0;
    this->oldSum = 0;
    this->newSum = 0;
    this->count = 0;
    this->newest = 0;
    this->stdevSum = 0;
    this->_min = std::numeric_limits<double>::infinity();
    this->_max = -std::numeric_limits<double>::infinity();
  }

};

}
#endif
