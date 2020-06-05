#include "../../include/Model.hpp"
#include "../../include/Tensor.hpp"
#include "../../include/ModelInfos.hpp"

#include "../../include/ModelManager.hpp"
#include "../../include/Loader.hpp"
#include "../../include/LoaderHarness.hpp"

#include "../../include/TFLoader.hpp"
#include "../../include/BasicModelManager.hpp"


#include <algorithm>
#include <iterator>


using namespace std;
using namespace ML;


template <typename Enumeration>
auto as_integer(Enumeration const value)
    -> typename std::underlying_type<Enumeration>::type
{
    return static_cast<typename std::underlying_type<Enumeration>::type>(value);
}

int main() {


    // Create model
    //Model m("../frozen_graph.pb");
    ////m.restore("../checkpoint/train.ckpt");

    ModelInfos<float> mInfos("../ModelInfo.txt");

    mInfos.parse();

    mInfos.show();




    //std::unique_ptr<Loader> p_loader;

    //p_loader =  make_unique<BasicLoader>("../frozen_graph.pb");

    std::array<ModelId,1> mlist = { ModelId{"mpg",{1,0}}};

    //auto p_loader_harness = make_unique<LoaderHarness>(model_id , std::move(p_loader));

    //p_loader_harness->Load();

    //cout << "state:" << as_integer(p_loader_harness->state())<<endl;

    //p_loader_harness->Unload();
    //p_loader->Unload();
    //cout << "state:" << as_integer(p_loader_harness->state())<<endl;

    std::unique_ptr<BasicModelManager> manager_;
    BasicModelManager::Create(&manager_);

    for (auto &el :  mlist){
        manager_->LoadModel(el,"../","frozen_graph.pb");
    }

    auto v = manager_->ListAvailableModelIds();

    for( auto &elem: v)
    {
        cout << "name : " << elem.name << "version:" << elem.version.major <<endl;
    }

    //manager_->UnloadModel(mlist[0]);

    //std::unique_ptr<UntypedModelHandle> untyped_handle ;
    
    ModelHandle<int64_t>  handle ; 
    manager_->GetModelHandle(mlist[0],&handle);


    manager_->UnloadModel(mlist[0]);

    



#if 0

    mInfos.compute_columns();

    std::cout << "depth:" <<mInfos.get_depth() <<std::endl;
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


#endif

    
}
