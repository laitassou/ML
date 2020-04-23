#include "../../include/Model.h"
#include "../../include/Tensor.h"
#include "../../include/ModelInfos.h"

#include <algorithm>
#include <iterator>


using namespace std;

int main() {


    // Create model
    Model m("../frozen_graph.pb");
    //m.restore("../checkpoint/train.ckpt");

    ModelInfos<float> mInfos("../ModelInfo.txt");
    mInfos.show();

    mInfos.parse();

    // Create Tensors
    Tensor input(m, "x");
    Tensor prediction(m, "Identity");


    std::array<float,7> mean {5.477,195.32,104.86,2990.25,15.56,75.89,1.573};
  
    std::array<float,7> std {1.699,104.331589 ,38.096214,843.898596,2.789230,3.675642,0.800988}; 
     /*
rain_stats:
              count         mean         std     min      25%     50%      75%     max
Cylinders     314.0     5.477707    1.699788     3.0     4.00     4.0     8.00     8.0
Displacement  314.0   195.318471  104.331589    68.0   105.50   151.0   265.75   455.0
Horsepower    314.0   104.869427   38.096214    46.0    76.25    94.5   128.00   225.0
Weight        314.0  2990.251592  843.898596  1649.0  2256.50  2822.5  3608.00  5140.0
Acceleration  314.0    15.559236    2.789230     8.0    13.80    15.5    17.20    24.8
Model Year    314.0    75.898089    3.675642    70.0    73.00    76.0    79.00    82.0
Origin        314.0     1.573248    0.800988     1.0     1.00     1.0     2.00     3.0

*/
    //std::vector<float> data {1.483887,1.865988,2.23462,1.018782,-2.530891,-1.604642,-0.715676};
    //std::vector<float> data {-0.869348  , -0.789008  , -0.259066 ,-0.903250, -0.559020 ,-1.332580 ,1.781239};
    //std::vector<float> data { 8,   318.0,      210.0,      4382,      13.5,   70 , 1};
    std::vector<float> data {4,   116.0 ,     90.00   ,   2123,    14.0 ,  71 , 2};

     std::vector<float> norm_data;
    cout << "data:" <<endl;

    auto i = 0;
    for (auto d:data)
    {
        auto normed = (d - mean[i]) / std[i];
        norm_data.push_back(normed);
        cout << "data:" <<d << "normed: "<<normed <<endl;
        i++;
    }

    // Feed data to input tensor
    input.set_data(norm_data);

    // Run and show predictions
    m.run(input, prediction);

    // Get tensor with predictions
    auto result = prediction.Tensor::get_data<float>();

     for (float f :result) {
        std::cout << f << " ";
    }
    std::cout << std::endl;


    
}
