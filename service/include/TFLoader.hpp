#pragma once


#include "Model.hpp"
#include "Loader.hpp"

namespace ML {

// Loader implemntation    

class BasicLoader :  public Loader {

public: 

    BasicLoader(const std::string & resource_path, const std::string & resource_name) : _resource_path(resource_path),
    _resource_name(resource_name) {}

    ~BasicLoader() { 
        cout << "basic loader destructor " << endl;
        //doUnload();
    }

    void * servable() { return reinterpret_cast<void *> (_servable.get());}

 private:    

    void doLoad() override 
    {
        //T  _m = new (_m) T(_resource_name);
        cout << "doLoad basic loader"<< endl;
        //return _m;
        _servable.reset( new Model(_resource_path+_resource_name));
    }
        
    void doUnload() override
    {
        cout << "doUnLoad basic loader"<< endl;
        _servable.reset();

        //delete m;
    }

    void doEstimate() override
    {}

    //T * _m;
    std::string _resource_path;
    std::string _resource_name;

    std::shared_ptr<Model> _servable;
    ModelHandle<Model> _handle;


};

}