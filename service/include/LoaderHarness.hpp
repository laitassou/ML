#pragma once


#include <memory>
#include <mutex>


#include "ModelManager.hpp"
#include "Loader.hpp"


namespace ML {


class LoaderHarness  {
 public:

  LoaderHarness(const ModelId& id, std::unique_ptr<Loader> loader);


  ~LoaderHarness();

  ModelId id() const { return _id; }

  State state() const ;

  Loader* loader() const { return _loader.get(); }


  Status Load();
  Status Unload();
  Status status() const;


  using  mutex_lock = std::lock_guard<std::mutex>;


 private:

  const ModelId _id;
  const std::shared_ptr<Loader> _loader;
  State _state = State::New;
  mutable std::mutex _mu;

  Status TransitionState(State from, State to);

};

}  
