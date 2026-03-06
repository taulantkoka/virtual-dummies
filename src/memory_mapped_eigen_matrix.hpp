/**
 * @file memory_mapped_eigen_matrix.hpp
 *
 * @brief Cross-platform memory-mapped matrix supporting the Eigen C++ library.
 *
 * Provides a template-based memory-mapped matrix class for use with Eigen
 * interface. Supports out-of-core data access for large datasets, and supports
 * multiple operating systems such as Windows, Linux, and macOS.
 * It supports arithmetic types (char, short, int, float, double) and complex
 * data types (std::complex<float>, std::complex<double>).
 *
 */

#ifndef MEMORY_MAPPED_EIGEN_MATRIX_HPP
#define MEMORY_MAPPED_EIGEN_MATRIX_HPP

// ============================================================
// Platform-Independent Standard Library Headers
// ============================================================
#include <chrono>
#include <complex>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>

// ============================================================
// Platform-Independent Third-Party Headers
// ============================================================
#include <Eigen/Dense>


// ============================================================
// Platform-Specific System Headers
// ============================================================
#ifdef _WIN32
// Windows: Use CreateFileMapping + MapViewofFile
#include <windows.h>
#else
// Unix/Linux/macOS: Use mmap
#include <cstring>    // strerror
#include <fcntl.h>    // open(), O_RDONLY, O_RDWR, O_CREAT
#include <sys/mman.h> // mmap(), munmap(), MAP_PRIVATE, MAP_SHARED, ...
#include <sys/stat.h> // fstat(), struct stat
#include <unistd.h>   // close(), ftruncante()
#endif


/**
 * @brief Template-based memory-mapped matrix for cross-platform out-of-core
 *  computing.
 *
 * This class provides a high-performance interface for working with
 * large matrices that may not fit in RAM.
 * It uses memory-mapped files to allow the operating system to manage data
 * movement between disk and memory efficiently.
 *
 * Supported types: char, short, int, float, double,
 *                  std::complex<float>, std::complex<double>
 *
 * Example usage:
 * @code
 * /// Create and write
 * MemoryMappedEigenMatrix<double> mat("data.bin", 10000, 5000,
 *                                MemoryMappedEigenMatrix<double>::Mode::ReadWrite);
 * auto map = mat.get_map_rw();
 * map(0, 0) = 42.0;
 *
 * /// Read and compute
 * MemoryMappedEigenMatrix<double> mat_read("data.bin", 10000, 5000);
 * auto view = mat_read.get_map();
 * double sum = view.sum();
 * @endcode
 *
 * @tparam T Element type (must be arithmetic or std::complex<float/double>)
 *
 * @note By default, the methods `get_map()` and `get_map_rw()` return an
 * Eigen matrix in column-major order.
 */
template <typename T>
class MemoryMappedEigenMatrix {

    // Type validation at compile time
    static_assert(std::is_arithmetic<T>::value ||
                      std::is_same<T, std::complex<float>>::value ||
                      std::is_same<T, std::complex<double>>::value,
                  "T must be an arithmetic type or std::complex<float/double>");

public:
    /**
     * @brief Access mode for memory-mapped file
     */
    enum class Mode {
        ReadOnly, // Read-only access (MAP_PRIVATE on Unix, FILE_MAP_READ on Windows)
        ReadWrite // Read-write access (MAP_SHARED on Unix, FILE_MAP_WRITE on Windows)
    };
    using Scalar = T; // Element type

private:
#ifdef _WIN32
    HANDLE hFile;      // Windows file handle
    HANDLE hMapping;   // Windows file mapping handle
    void *mapped_data; // Pointer to mapped memory
#else
    int fd;                // Unix file descriptor
    void *mapped_data;     // Pointer to mapped memory
    std::size_t file_size; // Size of mapped region in bytes
#endif
    std::size_t nrows_; // Number of matrix rows
    std::size_t ncols_; // Number of matrix columns
    Mode mode_;         // Access mode

public:
    /**
     * @brief Construct a new Memory-Mapped Matrix
     *
     * For ReadWrite mode, creates the file if its doesn't exist and resizes it
     * to the required size. For ReadOnly mode, opens an existing file.
     *
     * @param filename Path to the binary file.
     * @param nrows Number of rows in the matrix.
     * @param ncols Number of columns in the matrix.
     * @param mode Access mode (ReadOnly or ReadWrite)
     *
     * @throws std::run_time_error if file operations fail.
     */
    MemoryMappedEigenMatrix(const std::string &filename, std::size_t nrows,
                            std::size_t ncols, Mode mode = Mode::ReadOnly);

    /**
     * @brief Destructor - unmaps memory and closes file handles
     */
    ~MemoryMappedEigenMatrix();

    // Delete copy constructor and copy assignment (RAII resource management)
    MemoryMappedEigenMatrix(const MemoryMappedEigenMatrix &) = delete;
    MemoryMappedEigenMatrix &operator=(const MemoryMappedEigenMatrix &) =
        delete;

    /**
     * @brief Get a read-only Eigen::Map view of the mapped data.
     *
     * Returns an Eigen::Map with user-selected storage order, defaulting to
     * column-major.
     *
     * @tparam StorageOrder Eigen::ColMajor (default) or Eigen::RowMajor.
     * @return Eigen::Map<const Eigen::Matrix<T, ...>> for given order.
     *
     * @example
     * auto mat = mmap.get_map<>();                       // Default: ColMajor
     * auto mat_row = mmap.get_map<Eigen::RowMajor>();    // RowMajor
     */
    template <int StorageOrder = Eigen::ColMajor>
    Eigen::Map<
        const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, StorageOrder>>
    get_map() const;


    /**
     * @brief Get a read-write Eigen::Map view of the mapped data (only in
     *  Write mode).
     *
     * Returns an Eigen::Map with user-selected storage order, defaulting to
     *  column-major.
     *
     * @tparam StorageOrder Eigen::ColMajor (default) or Eigen::RowMajor.
     * @return Eigen::Map<Eigen::Matrix<T, ...>> for given order.
     * @throws std::runtime_error if called on a ReadOnly matrix.
     * @note Write Mode is not Thread Safe; use your own synchronization if
     *  accessed concurrently.
     *
     * @example
     * auto mat_rw = mmap.get_map_rw<>();                  // Default: ColMajor
     * auto mat_rw_row = mmap.get_map_rw<Eigen::RowMajor>();  // RowMajor
     */
    template <int StorageOrder = Eigen::ColMajor>
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, StorageOrder>>
    get_map_rw();

    /**
     * @brief Create a memory-mapped matrix file from a raw contiguous memory.
     *
     * Generic method that should work with most math libraries including:
     * Eigen, Armadillo, xtensor, and raw arrays. Requirement is a that data
     * are stored contiguously.
     *
     * @param filename Path where the binary file will be created.
     * @param data_ptr Pointe to contiguous matrix data.
     * @param nrows Number of rows.
     * @param ncols Number of columns.
     * @param is_column_major True (default) if data is column-major,
     *  false for row-major.
     * @param return_mode Access mode for returning mapping.
     *
     * @return MemoryMappedEigenMatrix object providing a view of the file.
     *
     * @throws std::runtime_error if operation fail.
     *
     * @note Ensure data_ptr points to (nrows * ncols * sizeof(T)) bytes.
     * @note Data layout (row/column-major) must match the access pattern used.
     *
     * @example
     * /// From raw array
     * double* my_data = new double[1000 * 500];
     * auto mmap = MemoryMappedEigenMatrix<double>::create_from_ptr(
     *  "data.bin", my_data, 1000, 500, true);
     *
     * /// From Armadillo
     * arma::mat X(1000, 500);
     * auto mmap = MemoryMappedEigenMatrix<double>::create_from_ptr(
     *  "arma.bin", X.memptr(), X.nrows, X.ncols, true);
     *
     * /// From Eigen
     * Eigen::MatrixXd Y(1000, 500);
     * auto mmap = MemoryMappedEigenMatrix<double>::create_from_ptr(
     *  "eigen.bin", Y.data(), Y.rows(), Y.cols(), true);
     */
    static MemoryMappedEigenMatrix<T> create_from_ptr(
        const std::string &filename, const T *data_ptr, std::size_t nrows,
        std::size_t ncols, bool is_column_major = true,
        Mode return_mode = Mode::ReadOnly);


    // Accessors methods
    /* Get the number of rows */
    std::size_t nrows() const {
        return nrows_;
    }
    /* Get the number of columns */
    std::size_t ncols() const {
        return ncols_;
    }
    /* Get raw pointer to mapped data */
    const void *data() const {
        return mapped_data;
    }
    /* Get access mode */
    Mode mode() const { return mode_; }

#ifdef _WIN32
    int fileno() const { return -1; }
#else
    int fileno() const { return fd; }
#endif

    /**
     * @brief Get file size in bytes
     *
     * @return Total size of mapped file
     */
    std::size_t size_bytes() const {
        return nrows_ * ncols_ * sizeof(T);
    }

    /**
     * @brief Get type name as string
     *
     * @return Type name as string
     */
    static constexpr const char *type_name();
};

// Convenience Type Aliases
// ============================================================
using MemMappedEigenMatrixChar = MemoryMappedEigenMatrix<char>;
using MemMappedEigenMatrixShort = MemoryMappedEigenMatrix<short>;
using MemMappedEigenMatrixInt = MemoryMappedEigenMatrix<int>;
using MemMappedEigenMatrixFloat = MemoryMappedEigenMatrix<float>;
using MemMappedEigenMatrixDouble = MemoryMappedEigenMatrix<double>;
using MemMappedEigenMatrixCFloat = MemoryMappedEigenMatrix<std::complex<float>>;
using MemMappedEigenMatrixCDouble =
    MemoryMappedEigenMatrix<std::complex<double>>;


// Implementation
// ============================================================
template <typename T>
MemoryMappedEigenMatrix<T>::MemoryMappedEigenMatrix(const std::string &filename,
                                                    std::size_t nrows,
                                                    std::size_t ncols,
                                                    Mode mode)
    : nrows_(nrows), ncols_(ncols), mode_(mode) {
#ifdef _WIN32
    // Windows implementation
    DWORD access = (mode == Mode::ReadOnly) ? GENERIC_READ
                                            : (GENERIC_READ | GENERIC_WRITE);
    DWORD creation = (mode == Mode::ReadWrite) ? CREATE_ALWAYS : OPEN_EXISTING;
    DWORD protect = (mode == Mode::ReadOnly) ? PAGE_READONLY : PAGE_READWRITE;
    DWORD map_access =
        (mode == Mode::ReadOnly) ? FILE_MAP_READ : FILE_MAP_WRITE;

    hFile = CreateFileA(filename.c_str(), access, 0, NULL, creation,
                        FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("Failed to open/create file: " + filename);
    }

    ULONGLONG byte_size = static_cast<ULONGLONG>(nrows_) * ncols_ * sizeof(T);

    if (mode == Mode::ReadWrite) {
        LARGE_INTEGER size;
        size.QuadPart = byte_size;
        hMapping = CreateFileMappingA(hFile, NULL, protect, size.HighPart,
                                      size.LowPart, NULL);
    } else {
        hMapping = CreateFileMapping(hFile, NULL, protect, 0, 0, NULL);
    }

    if (!hMapping) {
        CloseHandle(hFile);
        throw std::runtime_error("Failed to create file mapping");
    }

    mapped_data = MapViewOfFile(hMapping, map_access, 0, 0, 0);
    if (!mapped_data) {
        CloseHandle(hMapping);
        CloseHandle(hFile);
        throw std::runtime_error("Failed to map view of file");
    }
#else
    // Unix/Linux/macOS
    int flags =
        (mode == Mode::ReadOnly) ? O_RDONLY : (O_RDWR | O_CREAT | O_TRUNC);

    int prot = (mode == Mode::ReadOnly) ? PROT_READ : (PROT_READ | PROT_WRITE);

    int map_flags = (mode == Mode::ReadOnly) ? MAP_PRIVATE : MAP_SHARED;

    fd = open(filename.c_str(), flags, 0666);
    if (fd == -1) {
        throw std::runtime_error("Failed to open/create file" + filename +
                                 std::string(strerror(errno)) +
                                 " (errno=" + std::to_string(errno) + ")");
    }

    file_size = nrows_ * ncols_ * sizeof(T);

    // For write mode, resize the file
    if (mode == Mode::ReadWrite) {
        if (ftruncate(fd, file_size) == -1) {
            close(fd);
            throw std::runtime_error("Failed to resize file with ftruncate: " +
                                     std::string(strerror(errno)));
        }
    }

    mapped_data = mmap(nullptr, file_size, prot, map_flags, fd, 0);
    if (mapped_data == MAP_FAILED) {
        close(fd);
        throw std::runtime_error("Failed to mmap file: " +
                                 std::string(strerror(errno)));
    }
#endif
}

// Destructor Implementation
template <typename T>
MemoryMappedEigenMatrix<T>::~MemoryMappedEigenMatrix() {
#ifdef _WIN32
    if (mapped_data) UnmapViewOfFile(mapped_data);
    if (hMapping) CloseHandle(hMapping);
    if (hFile) CloseHandle(hFile);
#else
    if (mapped_data != MAP_FAILED) {
        // Sync before unmapping if in write mode
        if (mode_ == Mode::ReadWrite) {
            msync(mapped_data, file_size, MS_SYNC);
        }
        munmap(mapped_data, file_size);
    }
    if (fd != -1) close(fd);
#endif
}


// get_map() implementation
template <typename T>
template <int StorageOrder>
Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, StorageOrder>>
MemoryMappedEigenMatrix<T>::get_map() const {
    return Eigen::Map<
        const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, StorageOrder>>(
        static_cast<const T *>(mapped_data), nrows_, ncols_);
}


// get_map_rw() implementation
template <typename T>
template <int StorageOrder>
Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, StorageOrder>>
MemoryMappedEigenMatrix<T>::get_map_rw() {
    if (mode_ != Mode::ReadWrite) {
        throw std::runtime_error("Cannot get writable map in ReadOnly mode");
    }
    return Eigen::Map<
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, StorageOrder>>(
        static_cast<T *>(mapped_data), nrows_, ncols_);
}


// type_name() implementation
template <typename T>
constexpr const char *MemoryMappedEigenMatrix<T>::type_name() {
    if (std::is_same<T, char>::value) return "char";
    if (std::is_same<T, short>::value) return "short";
    if (std::is_same<T, int>::value) return "int";
    if (std::is_same<T, float>::value) return "float";
    if (std::is_same<T, double>::value) return "double";
    if (std::is_same<T, std::complex<float>>::value) return "complex<float>";
    if (std::is_same<T, std::complex<double>>::value) return "complex<double>";
    return "unknown";
}


// create_from_ptr() implementation
template <typename T>
MemoryMappedEigenMatrix<T> MemoryMappedEigenMatrix<T>::create_from_ptr(
    const std::string &filename, const T *data_ptr, std::size_t nrows,
    std::size_t ncols, bool is_column_major,
    MemoryMappedEigenMatrix<T>::Mode return_mode) {

    if (!data_ptr) {
        throw std::runtime_error("Null pointer provided to create_from_ptr");
    }

    std::size_t byte_size{nrows * ncols * sizeof(T)};

    {
        MemoryMappedEigenMatrix<T> temp_mmap(filename, nrows, ncols,
                                             Mode::ReadWrite);
        void *mapped_ptr = const_cast<void *>(temp_mmap.data());

        // Direct memory copy
        std::memcpy(mapped_ptr, data_ptr, byte_size);

#ifndef _WIN32
        if (msync(mapped_ptr, byte_size, MS_SYNC) == -1) {
            throw std::runtime_error("Failed to sync data: " +
                                     std::string(strerror(errno)));
        }
#else
        if (!FlushViewOfFile(mapped_ptr, byte_size)) {
            throw std::runtime_error("Failed to flush data to disk");
        }
#endif
    } // temporary mmap destructor wipes obj

    return MemoryMappedEigenMatrix<T>(filename, nrows, ncols, return_mode);
}


#endif // MEMORY_MAPPED_EIGEN_MATRIX_HPP