
#include "LoaderHarness.hpp"


using namespace std;

namespace ML {


LoaderHarness::LoaderHarness(const ModelId& id,std::unique_ptr<Loader> loader) 
:_id(id),_loader(std::move(loader)){
  cout<< "LoaderHarness "<< endl;
}

LoaderHarness::~LoaderHarness() {
  mutex_lock l(_mu);
  if(_state == State::New || _state == State::Disabled ||
         _state == State::Error){
     //cout << "Model: " << _id << " state: " << _state << endl;
  }
}

State LoaderHarness::state() const {
  mutex_lock l(_mu);
  return _state;
}


Status LoaderHarness::Load() {
  {
    mutex_lock l(_mu);
    TransitionState(State::New, State::Loading);
  }


  _loader->Load();


  {
    mutex_lock l(_mu);
    TransitionState(State::Loading, State::Ready);

  }


  return Status::OK;
}



Status LoaderHarness::Unload() {
  {
    mutex_lock l(_mu);
    TransitionState(State::Ready, State::Unloading);    
  }

  _loader->Unload();

  {
    mutex_lock l(_mu);
    TransitionState(State::Unloading, State::Disabled);
  }

  return Status::OK;
}


Status LoaderHarness::TransitionState(const State from, const State to) {
  if (_state != from) {
    const Status error = Status::ERROR;
    return error;
  }
  _state = to;
  return Status::OK;
}


} 
