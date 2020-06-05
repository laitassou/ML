#pragma once

#include <iostream>
#include "ModelManager.hpp"

using namespace std;

namespace ML {

 
// Loader abstract interface
class Loader{
 public:

    Loader(){ 
        cout << "Loader"<< endl;
    }

    void Load(){
        doLoad();
    }

    void Unload() { 
        doUnload();
    }

    void EstimateRessource(){
        doEstimate();
    }

    Status LoadWithMetadata(const ModelId& model){
        doLoadWithMetadata(model);
        return  Status::OK;
    } 

    virtual ~Loader(){};

    virtual void * servable() = 0;
    
 private:
    virtual Status doLoadWithMetadata(const ModelId& model) {Load();  return Status::ERROR;};
    virtual void doLoad() =0;
    virtual void doUnload() =0;
    virtual void doEstimate() {};


};

}