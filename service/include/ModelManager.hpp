/*
 *
 *
 *
 *
 *
 */

#ifndef _ML_MODEL_MANAGER_
#define _ML_MODEL_MANAGER_

#include <iostream>
#include <string>

#include <map>
#include <vector>


using namespace std;

namespace ML {


enum class State {
New,
Loading,
Ready,
Unloading,
Disabled,
Error
};

enum class Status{
    OK, 
    ERROR    
};



enum class Version_SELECT
{
    FIRST,
    LAST,
};

struct Version{
  size_t major;
  size_t minor;
  Version(size_t mj, size_t mn): major(mj),minor(mn){}
};


struct  ModelId {
  std::string name;
  Version version;
};


class UntypedModelHandle {
 public:
  virtual ~UntypedModelHandle() = default;

  virtual const ModelId& id() const = 0;


  virtual void * servable() = 0;


};



template <typename T> class ModelHandle {

    public: 

    ModelHandle() = default;

    const ModelId& id() const { return _p_model->id(); }

    T& operator*() const { return *get(); }

    T* operator->() const { return get(); }

    T* get() const { return _p_model; }

    operator bool() const { return get() != nullptr; }

    private:
    friend class ModelManager;

    explicit ModelHandle(std::unique_ptr<UntypedModelHandle> untyped_handle)
      : _untyped_handle(std::move(untyped_handle))
      {
          _p_model = reinterpret_cast<T*>(_untyped_handle->servable());
      }


    T* _p_model = nullptr;
    std::unique_ptr<UntypedModelHandle> _untyped_handle;
};




template<class InputType, class OutputType>  class ModelContext 
{
    public:
    ModelContext (size_t id,uint8_t inf_mask,size_t inp_size, size_t out_size, InputType * in_buf, OutputType * out_buf);

    private:        
    size_t id;
    uint8_t inference_mask;

    //InputType input_type;
    size_t input_size;
    //OutputType output_type;
    size_t output_size;
    //void * input_buffer;
    InputType * input_buffer;
    //void * output_buffer;
    OutputType * output_buffer;
};



template <typename T, typename U>
constexpr bool operator==(const ModelHandle<T>& l,
                          const ModelHandle<U>& r) {
  return l.get() == r.get() && l.id() == r.id();
}

template <typename T, typename U>
constexpr bool operator!=(const ModelHandle<T>& l,
                          const ModelHandle<U>& r) {
  return !(l == r);
}



inline bool operator==(const ModelId& l, const ModelId& r)
{
  return (l.name.compare( r.name)==0) && 
    l.version.major == r.version.major &&  
    l.version.minor == r.version.minor;
}



/*
 *
 *
 *
 *
 */
class ModelManager {
 public:
  virtual ~ModelManager() = default;

  virtual std::vector<ModelId> ListAvailableModelIds() const = 0;
  template <typename T> std::map<ModelId, ModelHandle<T>> GetAvailableModelHandles() const;
  template <typename T> Status GetModelHandle(const ModelId& request,ModelHandle<T>* const handle);

 private:

  virtual Status GetUntypedModelHandle(const ModelId& request, std::unique_ptr<UntypedModelHandle>* untyped_handle) = 0;
  virtual std::map<ModelId, std::unique_ptr<UntypedModelHandle>>  GetAvailableUntypedModelHandles() const = 0;
};


template <typename T>
Status ModelManager::GetModelHandle(const ModelId& request,ModelHandle<T>* const handle) {

  cout << "getHandle" << endl;
  std::unique_ptr<UntypedModelHandle> untyped_handle;
  GetUntypedModelHandle(request, &untyped_handle);
  if (untyped_handle == nullptr) {
      cout << "getHandle error" << endl;

    return Status::ERROR;
  }
  *handle = ModelHandle<T>(std::move(untyped_handle));
  if (handle->get() == nullptr) {
          cout << "getHandle nullptr" << endl;
    return Status::ERROR;
  }
  return Status::OK;
}


template <typename T>
std::map<ModelId, ModelHandle<T>> ModelManager::GetAvailableModelHandles()
    const {
  std::map<ModelId, ModelHandle<T>> id_and_handles;
  std::map<ModelId, std::unique_ptr<UntypedModelHandle>>
      id_and_untyped_handles = GetAvailableUntypedModelHandles();
  for (auto& id_and_untyped_handle : id_and_untyped_handles) {
    auto handle = ModelHandle<T>(std::move(id_and_untyped_handle.second));
    if (handle.get() != nullptr) {
      id_and_handles.emplace(id_and_untyped_handle.first, std::move(handle));
    }
  }
  return id_and_handles;
}


}

#endif //_ML_MODEL_MANAGER_