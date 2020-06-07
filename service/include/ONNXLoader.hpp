#pragma once


#include "Model.hpp"
#include "Loader.hpp"

namespace ML {

// Loader implemntation    
class ONNXMODEL {};

class ONNXLoader :  public Loader {

public: 

    ONNXLoader(const std::string & resource_path, const std::string & resource_name) : _resource_path(resource_path),
    _resource_name(resource_name) {}

    ~ONNXLoader() { 
        cout << "basic loader destructor " << endl;
        //doUnload();
    }

    void * servable() { return reinterpret_cast<void *> (_servable.get());}

 private:    

    void doLoad() override 
    {
        //T  _m = new (_m) T(_resource_name);
        cout << "doLoad ONNX loader"<< endl;
        //return _m;
        //_servable.reset( new Model(_resource_path+_resource_name));
    }
        
    void doUnload() override
    {
        cout << "doUnLoad ONNX loader"<< endl;
        _servable.reset();

        //delete m;
    }

    void doEstimate() override
    {}

    //T * _m;
    std::string _resource_path;
    std::string _resource_name;

    std::unique_ptr<ONNXMODEL> _servable;
    //ModelHandle<ONNXCLASS> _handle;


};

}