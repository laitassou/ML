#pragma once 

#include "LoaderHarness.hpp"
#include "ModelManager.hpp"

namespace ML {



class SharedPtrHandle final : public UntypedModelHandle {
 public:
  ~SharedPtrHandle() override = default;

  explicit SharedPtrHandle(const ModelId& id, std::shared_ptr<Loader> loader)
      : id_(id), loader_(std::move(loader)) {}

  void * servable() override { return loader_->servable(); }

  const ModelId& id() const override { return id_; }

 private:
  const ModelId id_;
  std::shared_ptr<Loader> loader_;
};


class BasicModelManager : public ModelManager {
 public:
  using PreHook = std::function<void(const uint32_t&)>;

  ~BasicModelManager() override { cout << "BasicModelManager destructor "<<endl;};

  std::vector<ModelId> ListAvailableModelIds() const override;


  void LoadModel(const ModelId& id,std::string path, std::string name);
  void UnloadModel(const ModelId& id);


  static Status Create(std::unique_ptr<BasicModelManager>* manager);


  private :

  struct Compare
  {
  
     bool operator() (const ModelId& l,
                              const ModelId& r)  const{
      return l.name.compare(r.name) && l.version.major == r.version.major && l.version.minor == r.version.minor ;
    }

  };

  using HandlesMap = std::map<ModelId,std::shared_ptr< LoaderHarness> ,Compare>;
  using  mutex_lock = std::lock_guard<std::mutex>;

  HandlesMap   handles_map_;


  Status GetUntypedModelHandle(const ModelId& request, std::unique_ptr<UntypedModelHandle>* untyped_handle) override ;

  std::map<ModelId, std::unique_ptr<UntypedModelHandle>>GetAvailableUntypedModelHandles() const override {};

  mutable mutex mu;


};

std::vector<ModelId> BasicModelManager::ListAvailableModelIds() const {
    std::vector<ModelId> ids;
  
    HandlesMap  handles_map = handles_map_;
    for (auto iter = handles_map.begin(); iter != handles_map.end(); iter++) {
      // We get the iterator where all the values for a particular key ends.

        if (iter->first.version.major>=1) {
         ids.push_back(iter->first);
        }
      
    }
    
    return ids;
}




Status BasicModelManager::GetUntypedModelHandle(const ModelId& request,std::unique_ptr<UntypedModelHandle>* const untyped_handle) {
  const auto found_it = handles_map_.find(request);
  if (found_it == handles_map_.end()) {
    return Status::ERROR;
  }
  //std::shared_ptr<const HandlesMap> handles_map = handles_map_.get();

  const LoaderHarness& harness = *found_it->second;

  untyped_handle->reset(new SharedPtrHandle(
      harness.id(), std::shared_ptr<Loader>(harness.loader())));

  return Status::OK;
}

/*
std::map<ModelId, std::unique_ptr<UntypedModelHandle>>BasicModelManager::GetAvailableUntypedModelHandles() const {
  std::map<ModelId, std::unique_ptr<UntypedModelHandle>> result;

  //std::shared_ptr<const HandlesMap> handles_map = handles_map_.get();

  for (const auto& handle : *handles_map_) {
    const ModelId& request = handle.first;
    if (!request.version) {
      continue;
    }
    const LoaderHarness& harness = *handle.second;
    result.emplace(harness.id(),
                   std::unique_ptr<UntypedModelHandle>(new SharedPtrHandle(
                       harness.id(), std::shared_ptr<Loader>(handles_map_,harness.loader()))));
  }
  return result;
}


*/


void BasicModelManager::LoadModel(const ModelId& id,std::string path, std::string name) {
    cout << " load servable " << id.name << " version:" <<id.version.major<<endl;
    mutex_lock l(mu);
    std::unique_ptr<Loader> p_loader;

    p_loader =  make_unique<BasicLoader>(path,name);

    std::shared_ptr<LoaderHarness>  p_loader_harness = make_shared<LoaderHarness>(id , std::move(p_loader));

    Status status = p_loader_harness->Load();

    if( status == Status::OK){
      handles_map_.emplace(id,p_loader_harness);
      cout << "size:" <<handles_map_.size()<<endl;
      cout << " emplace model " << id.name << " version:" <<id.version.major<<endl;

    }
    else
    {
      cout << "failed to load model :"<< id.name << endl;
    }
}


void BasicModelManager::UnloadModel(const ModelId& model) {
    //cout << " unload model " << model.name << " version:" <<model.version.major<<"." <<model.version.minor<<endl;
    const auto found_it = handles_map_.find(model);

    mutex_lock l(mu);
    if(found_it != handles_map_.end()){
      handles_map_.erase(found_it);
      cout << "UnloadModel size:" <<handles_map_.size()<<endl;

    }
    else
    {
      cout << "nout found" << endl;
    }
}


Status BasicModelManager::Create(std::unique_ptr<BasicModelManager>* manager) {
  manager->reset(new BasicModelManager());
  return Status::OK;
}


}