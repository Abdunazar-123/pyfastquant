#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "engine/monte_carlo.hpp"
#include "engine/black_scholes.hpp"
#include "engine/linear_algebra.hpp"
#include "engine/simd_math.hpp"

#include <memory>
#include <thread>

namespace py = pybind11;
using namespace PyFastQuant::Engine;


class PyMonteCarloEngine {
private:
    std::unique_ptr<MonteCarloEngine> m_engine;

public:
    PyMonteCarloEngine(std::size_t num_paths, std::size_t num_time_steps, std::size_t threads)
        : m_engine(std::make_unique<MonteCarloEngine>(num_paths, num_time_steps, threads)) {}

    py::dict price_european_option(
        double S0, double K, double T, double r, double sigma,
        bool is_call, bool use_antithetic, double confidence_level
    ) {
        py::gil_scoped_release release;
        auto result = m_engine->price_european_option(S0, K, T, r, sigma, is_call, use_antithetic, confidence_level);
        py::gil_scoped_acquire acquire;

        py::dict output;
        output["price"] = result.price;
        output["std_error"] = result.std_error;
        output["confidence_interval_low"] = result.confidence_interval_low;
        output["confidence_interval_high"] = result.confidence_interval_high;
        output["computation_time_ms"] = result.computation_time_ms;
        output["used_antithetic"] = result.used_antithetic;
        return output;
    }

    py::list price_european_option_batch(
        py::array_t<double> S0,
        py::array_t<double> K,
        py::array_t<double> T,
        py::array_t<double> r,
        py::array_t<double> sigma,
        py::array_t<bool> is_call,
        bool use_antithetic
    ) {
        auto S0_info = S0.request();
        std::size_t n = static_cast<std::size_t>(S0_info.size);

        double* S0_ptr = static_cast<double*>(S0_info.ptr);
        double* K_ptr = static_cast<double*>(K.request().ptr);
        double* T_ptr = static_cast<double*>(T.request().ptr);
        double* r_ptr = static_cast<double*>(r.request().ptr);
        double* sigma_ptr = static_cast<double*>(sigma.request().ptr);
        bool* is_call_ptr = static_cast<bool*>(is_call.request().ptr);

        std::vector<double> S0_vec(S0_ptr, S0_ptr + n);
        std::vector<double> K_vec(K_ptr, K_ptr + n);
        std::vector<double> T_vec(T_ptr, T_ptr + n);
        std::vector<double> r_vec(r_ptr, r_ptr + n);
        std::vector<double> sigma_vec(sigma_ptr, sigma_ptr + n);
        std::vector<bool> is_call_vec(is_call_ptr, is_call_ptr + n);

        py::gil_scoped_release release;
        auto results = m_engine->price_european_option_batch(
            S0_vec, K_vec, T_vec, r_vec, sigma_vec, is_call_vec, use_antithetic
        );
        py::gil_scoped_acquire acquire;

        py::list output;
        for (const auto& result : results) {
            py::dict d;
            d["price"] = result.price;
            d["std_error"] = result.std_error;
            d["confidence_interval_low"] = result.confidence_interval_low;
            d["confidence_interval_high"] = result.confidence_interval_high;
            d["computation_time_ms"] = result.computation_time_ms;
            d["used_antithetic"] = result.used_antithetic;
            output.append(d);
        }
        return output;
    }

    std::size_t get_num_paths() const { return m_engine->m_num_paths; }
    std::size_t get_num_time_steps() const { return m_engine->m_num_time_steps; }
    std::size_t get_threads() const { return m_engine->m_threads; }
};


class PyBlackScholesEngine {
public:
    static py::dict price_option(
        double S0, double K, double T, double r, double sigma, bool is_call
    ) {
        py::gil_scoped_release release;
        auto result = BlackScholesEngine::price_option(S0, K, T, r, sigma, is_call);
        py::gil_scoped_acquire acquire;

        py::dict output;
        output["price"] = result.price;
        output["delta"] = result.delta;
        output["gamma"] = result.gamma;
        output["theta"] = result.theta;
        output["vega"] = result.vega;
        output["rho"] = result.rho;
        return output;
    }
};


class PyLinearAlgebra {
public:
    static py::array_t<double> matrix_multiply(
        py::array_t<double> A, py::array_t<double> B
    ) {
        auto A_info = A.request();
        auto B_info = B.request();

        if (A_info.ndim != 2 || B_info.ndim != 2) {
            throw std::runtime_error("Both inputs must be 2D arrays");
        }

        std::size_t rowsA = static_cast<std::size_t>(A_info.shape[0]);
        std::size_t colsA = static_cast<std::size_t>(A_info.shape[1]);
        std::size_t rowsB = static_cast<std::size_t>(B_info.shape[0]);
        std::size_t colsB = static_cast<std::size_t>(B_info.shape[1]);

        if (colsA != rowsB) {
            throw std::runtime_error("Matrix dimensions incompatible for multiplication");
        }

        auto result = py::array_t<double>({rowsA, colsB});
        auto result_info = result.request();

        double* A_ptr = static_cast<double*>(A_info.ptr);
        double* B_ptr = static_cast<double*>(B_info.ptr);
        double* C_ptr = static_cast<double*>(result_info.ptr);

        py::gil_scoped_release release;
        LinearAlgebraEngine::matrix_multiply_simd(A_ptr, B_ptr, C_ptr, rowsA, colsA, colsB);
        py::gil_scoped_acquire acquire;

        return result;
    }

    static py::array_t<double> cholesky(py::array_t<double> matrix) {
        auto info = matrix.request();
        if (info.ndim != 2 || info.shape[0] != info.shape[1]) {
            throw std::runtime_error("Cholesky requires a square 2D matrix");
        }
        std::size_t n = static_cast<std::size_t>(info.shape[0]);

        auto result = py::array_t<double>({n, n});
        auto result_info = result.request();
        std::memcpy(result_info.ptr, info.ptr, n * n * sizeof(double));

        py::gil_scoped_release release;
        LinearAlgebraEngine::cholesky_decomposition(static_cast<double*>(result_info.ptr), n);
        py::gil_scoped_acquire acquire;

        return result;
    }
};


PYBIND11_MODULE(_core, m) {
    m.doc() = "PyFastQuant - high-performance quantitative finance library "
              "(C++ engine + AVX2 SIMD + OpenMP, exposed via pybind11)";

    py::class_<PyMonteCarloEngine>(m, "MonteCarloEngine")
        .def(py::init<std::size_t, std::size_t, std::size_t>(),
             py::arg("num_paths") = 1000000,
             py::arg("num_time_steps") = 100,
             py::arg("threads") = 0,
             "Initialize the Monte Carlo engine. threads=0 means "
             "'use all available hardware threads'.")
        .def("price_european_option", &PyMonteCarloEngine::price_european_option,
             py::arg("S0"), py::arg("K"), py::arg("T"),
             py::arg("r"), py::arg("sigma"), py::arg("is_call"),
             py::arg("use_antithetic") = false,
             py::arg("confidence_level") = 0.95,
             "Price a single European option. Set use_antithetic=True "
             "for variance-reduced pricing at the same path count.")
        .def("price_european_option_batch", &PyMonteCarloEngine::price_european_option_batch,
             py::arg("S0"), py::arg("K"), py::arg("T"),
             py::arg("r"), py::arg("sigma"), py::arg("is_call"),
             py::arg("use_antithetic") = false)
        .def_property_readonly("num_paths", &PyMonteCarloEngine::get_num_paths)
        .def_property_readonly("num_time_steps", &PyMonteCarloEngine::get_num_time_steps)
        .def_property_readonly("threads", &PyMonteCarloEngine::get_threads);

    py::class_<PyBlackScholesEngine>(m, "BlackScholesEngine")
        .def_static("price_option", &PyBlackScholesEngine::price_option,
             py::arg("S0"), py::arg("K"), py::arg("T"),
             py::arg("r"), py::arg("sigma"), py::arg("is_call"));

    py::class_<PyLinearAlgebra>(m, "LinearAlgebra")
        .def_static("matrix_multiply", &PyLinearAlgebra::matrix_multiply)
        .def_static("cholesky", &PyLinearAlgebra::cholesky);

    m.attr("__version__") = "2.0.0";

    m.def("get_cache_line_size", []() { return CACHE_LINE_SIZE; });
    m.def("get_hardware_concurrency", []() {
        auto n = std::thread::hardware_concurrency();
        return n == 0 ? std::size_t(1) : std::size_t(n);
    });
    m.def("has_avx2", []() { return static_cast<bool>(PFQ_HAS_AVX2); });
}
