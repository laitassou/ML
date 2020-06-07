
#include "BasicModelManager.hpp"


using namespace std;

namespace ML {

std::vector<ModelId> BasicModelManager::ListAvailableModelIds() const 
{
    std::vector<ModelId> ids;
  
    HandlesMap * handles_map = handles_map_.get();
    for (auto iter = handles_map->begin(); iter != handles_map->end(); iter++) {
        if (iter->first.version.major>=1) {
         ids.push_back(iter->first);
        }     
    }
    
    return ids;
}


Status BasicModelManager::GetUntypedModelHandle(const ModelId& request,std::unique_ptr<UntypedModelHandle>* const untyped_handle)
{
 
  const auto found_it = handles_map_->find(request);
  if (found_it == handles_map_->end()) {
    return Status::ERROR;
  }
  //std::shared_ptr<const HandlesMap> handles_map = handles_map_.get();

  const LoaderHarness& harness = *found_it->second;
  std::shared_ptr<HandlesMap> handles_map = handles_map_;

  untyped_handle->reset(new SharedPtrHandle(
      harness.id(), std::shared_ptr<Loader>(handles_map,harness.loader())));
 
  return Status::OK;
}


void BasicModelManager::LoadModel(const ModelId& id,std::string path, std::string name, ModelType type)
{
    cout << " load servable " << id.name << " version:" <<id.version.major<<endl;
    mutex_lock l(mu);
    std::unique_ptr<Loader> p_loader;

    const auto found_it = handles_map_->find(id);
    if(found_it == handles_map_->end())
    {

      if (type == ModelType::TFL)
          p_loader =  make_unique<TFLoader>(path,name);
      else if (type == ModelType::ONNX)
          p_loader =  make_unique<ONNXLoader>(path,name);
      else
          p_loader =  make_unique<MRVLLoader>(path,name);

      std::shared_ptr<LoaderHarness>  p_loader_harness = make_shared<LoaderHarness>(id , std::move(p_loader));

      Status status = p_loader_harness->Load();

      if( status == Status::OK){
        handles_map_->emplace(id,p_loader_harness);
        cout << "size:" <<handles_map_->size()<<endl;
        cout << " emplace model " << id.name << " version:" <<id.version.major<<endl;
      }
      else
      {
        cout << "failed to load model :"<< id.name << endl;
      }

    }else
    {
      cout << "model exists and loaded before :"<< id.name << endl;
    }
}


void BasicModelManager::UnloadModel(const ModelId& model)
{
    //cout << " unload model " << model.name << " version:" <<model.version.major<<"." <<model.version.minor<<endl;
    const auto found_it = handles_map_->find(model);

    mutex_lock l(mu);
    if(found_it != handles_map_->end()){
      handles_map_->erase(found_it);
      cout << "UnloadModel size:" <<handles_map_->size()<<endl;

    }
    else
    {
      cout << "nout found" << endl;
    }
}


Status BasicModelManager::Create(std::unique_ptr<BasicModelManager>* manager){
  manager->reset(new BasicModelManager());
  return Status::OK;
}

}
