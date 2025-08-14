#include <memory>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <trigdx/trigdx.hpp>

namespace py = pybind11;

template <typename T>
py::array_t<T>
compute_sin(const Backend &backend,
            py::array_t<T, py::array::c_style | py::array::forcecast> x) {
  ssize_t n = x.shape(0);
  const T *x_ptr = x.data();

  py::array_t<float> s(n);
  T *s_ptr = s.mutable_data();

  backend.compute_sinf(static_cast<size_t>(n), x_ptr, s_ptr);

  return s;
}

template <typename T>
py::array_t<T>
compute_cos(const Backend &backend,
            py::array_t<T, py::array::c_style | py::array::forcecast> x) {
  ssize_t n = x.shape(0);
  const T *x_ptr = x.data();

  py::array_t<T> c(n);
  T *c_ptr = c.mutable_data();

  backend.compute_cosf(static_cast<size_t>(n), x_ptr, c_ptr);

  return c;
}

template <typename T>
std::tuple<py::array_t<T>, py::array_t<T>>
compute_sincos(const Backend &backend,
               py::array_t<T, py::array::c_style | py::array::forcecast> x) {
  ssize_t n = x.shape(0);
  const T *x_ptr = x.data();

  py::array_t<T> s(n);
  py::array_t<T> c(n);

  backend.compute_sincosf(static_cast<size_t>(n), x_ptr, s.mutable_data(),
                          c.mutable_data());

  return std::make_tuple(s, c);
}

template <typename BackendType>
void bind_backend(py::module &m, const char *name) {
  py::class_<BackendType, Backend, std::shared_ptr<BackendType>>(m, name)
      .def(py::init<>())
      .def("compute_sinf", &compute_sin<float>)
      .def("compute_cosf", &compute_cos<float>)
      .def("compute_sincosf", &compute_sincos<float>);
}

PYBIND11_MODULE(pytrigdx, m) {
  py::class_<Backend, std::shared_ptr<Backend>>(m, "Backend")
      .def("init", &Backend::init);

  bind_backend<ReferenceBackend>(m, "Reference");
  bind_backend<LookupBackend<16384>>(m, "Lookup16K");
  bind_backend<LookupBackend<32768>>(m, "Lookup32K");
  bind_backend<LookupAVXBackend<16384>>(m, "LookupAVX16K");
  bind_backend<LookupAVXBackend<32768>>(m, "LookupAVX32K");
#if defined(TRIGDX_USE_MKL)
  bind_backend<MKLBackend>(m, "MKL");
#endif
#if defined(TRIGDX_USE_GPU)
  bind_backend<GPUBackend>(m, "GPU");
#endif
#if defined(TRIGDX_USE_XSIMD)
  bind_backend<LookupXSIMDBackend<16384>>(m, "LookupXSIMD16K");
  bind_backend<LookupXSIMDBackend<32768>>(m, "LookupXSIMD32K");
#endif
}