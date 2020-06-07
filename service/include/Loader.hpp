/*
 *
 *
 *
 *
 */
#ifndef _ML_LOADER_
#define _ML_LOADER_

#include <iostream>
#include "ModelManager.hpp"

using namespace std;

namespace ML {

 
// Loader interface to load/unload model
// NVI interface to offer more flexibility to check preconditions/ postconditions
class Loader{
 public:

    Loader(){ 
        cout << "Loader ctor"<< endl;
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

    virtual ~Loader(){};
    virtual void * servable() = 0;
    
 private:
    virtual void doLoad() =0;
    virtual void doUnload() =0;
    virtual void doEstimate() {};


};

}

#endif //_ML_LOADER_