//
// Model
//

#pragma once

#include <cstring>
#include <algorithm>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <tuple>
#include <memory>
#include <tensorflow/c/c_api.h>
#include "Tensor.h"

class Tensor;

//using GraphResourcePtr = std::unique_ptr<TF_Graph> ;


class Model {
public:
    explicit Model(const std::string&);

    // Rule of five, moving is easy as the pointers can be copied, copying not as i have no idea how to copy
    // the contents of the pointer (i guess dereferencing won't do a deep copy)
    Model(const Model &model) = delete;
    Model(Model &&model) = default;
    Model& operator=(const Model &model) = delete;
    Model& operator=(Model &&model) = default;

    ~Model();

    void init();
    void restore(const std::string& ckpt);
    void save(const std::string& ckpt);
    std::vector<std::string> get_operations() const;

    // Original Run
    void run(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs);

    // Run with references
    void run(Tensor& input, const std::vector<Tensor*>& outputs);
    void run(const std::vector<Tensor*>& inputs, Tensor& output);
    void run(Tensor& input, Tensor& output);

    // Run with pointers
    void run(Tensor* input, const std::vector<Tensor*>& outputs);
    void run(const std::vector<Tensor*>& inputs, Tensor* output);
    void run(Tensor* input, Tensor* output);

    struct GraphCreate {
        TF_Graph * operator()() { return TF_NewGraph(); }
    };
    struct GraphDeleter {
        void operator()(TF_Graph* b) { TF_DeleteGraph(b); }
    };

private:
    std::unique_ptr<TF_Graph, GraphDeleter> graph;
    TF_Session* session;
    TF_Status* status;

    // Read a file from a string
    static TF_Buffer* read(const std::string&);

    bool status_check(bool throw_exc) const;
    void error_check(bool condition, const std::string &error) const;

public:
    friend class Tensor;
};


