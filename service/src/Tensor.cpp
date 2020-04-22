#include "../include/Tensor.h"

#include <utility>

Tensor::Tensor(const Model& model, const std::string& operation) {

    // Get operation by the name
    this->op.oper = TF_GraphOperationByName(model._graph.get(), operation.c_str());
    this->op.index = 0;

    // Operation did not exists
    error_check(this->op.oper != nullptr, "No operation named \"" + operation + "\" exists" );

    // DIMENSIONS

    // Get number of dimensions
    int n_dims = TF_GraphGetTensorNumDims(model._graph.get(), this->op, model._status.get());

    // DataType
    this->type = TF_OperationOutputType(this->op);

    // If is not a scalar
    if (n_dims > 0) {
        // Get dimensions
        auto *dims = new int64_t[n_dims];
        TF_GraphGetTensorShape(model._graph.get(), this->op, dims, n_dims, model._status.get());

        // Check error on Model Status
        model.status_check(true);

        this->shape = std::vector<int64_t>(dims, dims + n_dims);

        // Only one dimension can be unknown using this constructor
        // error_check(std::count(this->shape.begin(), this->shape.end(), -1) <= 1, "At most one dimension can be unknown");

        delete[] dims;
    }

    this->flag = 0;
    this->val = nullptr;
    this->data = nullptr;
}

Tensor::~Tensor() {
    this->clean();
}



void Tensor::clean() {
    if (this->flag == 1) {
        TF_DeleteTensor(this->val);
        this->flag = 0;
    }
    this->data = nullptr;
}

void  Tensor::error_check(bool condition, const std::string &error) {
    if (!condition) {
        this->flag = -1;
        throw std::runtime_error(error);
    }
}

template<typename T>
void Tensor::set_data(std::vector<T> new_data) {

    //Non empty tensor
    if (this->flag == 1) {
        TF_DeleteTensor(this->val);
        this->flag = 0;
    }

    // Check Tensor is valid
    this->error_check(this->flag != -1, "Tensor is not valid");

    // Check type
    this->error_check(deduce_type<T>() == this->type, "Provided type is different from Tensor expected type");

    // Dimensions must be known
    this->error_check(!this->shape.empty(), "Shape of the input Tensor is not known, please provide a shape");

    // At most one dimension can be unknown
    this->error_check(std::count(this->shape.begin(), this->shape.end(), -1) >= -1, "At most one dimension can be unknown, please provide a shape");

    // Check number of elements
    auto exp_size = std::abs(std::accumulate(this->shape.begin(), this->shape.end(), 1, std::multiplies<int64_t>()));

    this->error_check(new_data.size() % exp_size == 0, "Expected and provided number of elements do not match");

    // Deallocator
    auto d = [](void* ddata, size_t, void*) {free(static_cast<T*>(ddata));};


    // Calculate actual shape of unknown dimensions
    this->actual_shape = std::make_unique<decltype(actual_shape)::element_type>(shape.begin(), shape.end());
    std::replace_if (actual_shape->begin(), actual_shape->end(), [](int64_t r) {return r==-1;}, new_data.size()/exp_size);

    // Saves data on class
    this->data = malloc(sizeof(T) * new_data.size());
    memcpy(this->data, new_data.data(), sizeof(T) * new_data.size());

    this->val = TF_NewTensor(this->type, actual_shape->data(), actual_shape->size(), this->data, sizeof(T) * new_data.size(), d, nullptr);


    this->error_check(this->val != nullptr, "An error occurred allocating the Tensor memory");

    this->flag = 1;
}

template<typename T> void Tensor::set_data(std::vector<T> new_data, const std::vector<int64_t>& new_shape) {

    this->error_check(this->shape.empty() || this->shape.size() == new_shape.size(), "Provided shape has different number of dimensions");
    auto old_shape = this->shape;

    this->shape = new_shape;
    this->set_data(new_data);

    this->shape = old_shape;
}

template<typename T>
std::vector<T> Tensor::get_data() {

    // Check Tensor is valid
    this->error_check(this->flag != -1, "Tensor is not valid");

    // Check type
    this->error_check(deduce_type<T>() == this->type, "Expected return type is different from Tensor type");

    // Tensor is not empty
    this->error_check(this->flag != 0, "Tensor is empty");


    // Check tensor data is not empty
    auto raw_data = TF_TensorData(this->val);
    this->error_check(raw_data != nullptr, "Tensor data is empty");

    size_t size = TF_TensorByteSize(this->val) / TF_DataTypeSize(TF_TensorType(this->val));

    // Convert to correct type
    const auto T_data = static_cast<T*>(raw_data);
    return std::vector<T>(T_data, T_data + size);
}

std::vector<int64_t> Tensor::get_shape() {
	return shape;
}

template<typename T>
TF_DataType Tensor::deduce_type() {
    if (std::is_same<T, float>::value)
        return TF_FLOAT;
    if (std::is_same<T, double>::value)
        return TF_DOUBLE;

    throw std::runtime_error{"Could not deduce type!"};
}

void Tensor::deduce_shape() {
    // Get number of dimensions
    int n_dims = TF_NumDims(this->val);

    // If is not a scalar
    if (n_dims > 0) {
        // Get dimensions
        this->shape = std::vector<int64_t>(n_dims, -1);
        for (int i=0; i<n_dims; i++) {
            this->shape[i] = TF_Dim(this->val, i);
        }
    }
}


// VALID deduce_type TEMPLATES
template TF_DataType Tensor::deduce_type<float>();
template TF_DataType Tensor::deduce_type<double>();
//template TF_DataType Tensor::deduce_type<bool>();

// VALID get_data TEMPLATES
template std::vector<float> Tensor::get_data<float>();
template std::vector<double> Tensor::get_data<double>();
template std::vector<bool> Tensor::get_data<bool>();


// VALID set_data TEMPLATES
template void Tensor::set_data<float>(std::vector<float> new_data);
template void Tensor::set_data<double>(std::vector<double> new_data);
//template void Tensor::set_data<bool>(std::vector<bool> new_data);


// VALID set_data TEMPLATES
template void Tensor::set_data<float>(std::vector<float> new_data, const std::vector<int64_t>& new_shape);
template void Tensor::set_data<double>(std::vector<double> new_data, const std::vector<int64_t>& new_shape);
//template void Tensor::set_data<bool>(std::vector<bool> new_data, const std::vector<int64_t>& new_shape);
