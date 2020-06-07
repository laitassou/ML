#pragma once


#include "Model.hpp"
#include "Loader.hpp"

namespace ML {

// MRVL Loader implemntation
// Call api and add needed resources : package_id etc
class MRVLLoader :  public Loader {

public: 

    MRVLLoader(const std::string & resource_path, const std::string & resource_name) : _resource_path(resource_path),
    _resource_name(resource_name) {}

    ~MRVLLoader()
    { 
        cout << "MRVL loader destructor " << endl;
    }

    void * servable()
    { 
        return reinterpret_cast<void *> (_servable.get());
    }

 private:    

    void doLoad() override 
    {
        cout << "doLoad MRVL loader"<< endl;
        // send request to load MRVL package, model 
    }
        
    void doUnload() override
    {
        cout << "doUnLoad MRVL loader"<< endl;
        //_servable.reset();

    }

    void doEstimate() override
    {}

    //T * _m;
    std::string _resource_path;
    std::string _resource_name;

    std::unique_ptr<int> _servable;
    //ModelHandle<Model> _handle;


};

}