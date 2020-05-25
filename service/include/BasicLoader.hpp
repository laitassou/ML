#include "Model.hpp"
#include "Loader.hpp"

namespace ML {

template <class T > class BasicLoader : public  template <class T> Loader {

public: 

BasicLoader(std::string resource_name) _resource_name(resource_name) {};
 private:    

    T * doLoad() override 
    {
        T  _m = new (_m) Model(_resource_name);

        return _m;
    }
        
    void doUnload() override
    {

    }

    void doEstimate() override
    {}

    T * _m;

};

}