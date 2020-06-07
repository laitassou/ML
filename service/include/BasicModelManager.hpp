#pragma once 

#include "LoaderHarness.hpp"
#include "ModelManager.hpp"

//#define ONNX_MODEL
//#if defined(TENSORFLOW_MODEL)
#include "TFLoader.hpp"
//#elif defined(ONNX_MODEL)
#include "ONNXLoader.hpp"
//#else
#include "MRVLLoader.hpp"
//#endif 

namespace ML {



enum class  ModelType {
  MRVL,
  ONNX,
  TFL
};

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

  void LoadModel(const ModelId& id,std::string path, std::string name, ModelType type );
  void UnloadModel(const ModelId& id);

  static Status Create(std::unique_ptr<BasicModelManager>* manager);

  BasicModelManager(){
    handles_map_ = std::make_shared<HandlesMap>();
  }

  private :

  struct Compare
  {
     bool operator() (const ModelId& l,const ModelId& r)  const
     {
        return l.name.compare( r.name) <0 || 
        l.name.compare( r.name) ==0 && l.version.major <r.version.major ||
        l.name.compare( r.name) ==0 && l.version.major == r.version.major && l.version.minor < r.version.minor;
     }

  };

  using HandlesMap = std::map<ModelId,std::shared_ptr< LoaderHarness> , Compare>;
  using  mutex_lock = std::lock_guard<std::mutex>;

  std::shared_ptr<HandlesMap>   handles_map_;
  mutable mutex mu;

  Status GetUntypedModelHandle(const ModelId& request, std::unique_ptr<UntypedModelHandle>* untyped_handle) override ;

  std::map<ModelId, std::unique_ptr<UntypedModelHandle>>GetAvailableUntypedModelHandles() const override {};

};


}