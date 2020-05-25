#pragma once 

class BasicModelManager : public ModelManager {
 public:
  using PreHook = std::function<void(const uint32_t&)>;

  ~BasicModelManager();

}