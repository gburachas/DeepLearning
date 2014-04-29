#ifndef OCLMATRIX_H
#define OCLMATRIX_H

#include <algorithm>
#include <iterator>
#include <functional>
#include <math.h>
#include <numeric>
#include <assert.h>
#include <limits>
#include <string>
#include <stdio.h>
#include <sstream>
#include <time.h>
#include <chrono>

#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <ailab/common.h>
#include <ailab/opencl.h>
#include <ext/lodepng.h>

namespace ailab {

    /** ------------------------------ Member Classes ------------------------------ **/
    template<typename T>
    class OCLMatrix : public CLBacked {
    public:

        OCLMatrix(const OpenCL::spContext context = NULL)
        : dev(NULL),
        errMeasure(NULL),
        host(NULL),
        n_cells(0),
        size(0),
        hist_count(NULL),
        sp_host(NULL),
        sp_hist_count(NULL),
        CLBacked(context) {

            srand(time(NULL));
        }

        OCLMatrix<T>& operator=(const OCLMatrix<T>& rhs) {
            this->hist_count = NULL;
            this->dev = rhs.dev;
            this->host = rhs.host;
            this->n_cells = rhs.n_cells;
            this->size = rhs.size;
        }

        void init(OCLMatrix<T>& sameAs) {
            this->init(sameAs.dims);
        }

        void init(std::vector<size_t> dims, T* host_mem = NULL) {
            this->dims.insert(this->dims.end(), dims.begin(), dims.end());
            this->_init_dims.insert(this->_init_dims.end(), dims.begin(), dims.end());
            this->n_cells = std::accumulate(dims.begin(), dims.end(), 1,
                    std::multiplies<size_t>());

            this->size = this->n_cells * sizeof (T);

            if (this->size > 0) {
                if (this->context != NULL) {
                    this->dev = this->context->alloc(this->size, CL_MEM_READ_WRITE, host_mem);
                    this->errMeasure = this->context->alloc(sizeof (cl_uint), CL_MEM_READ_WRITE, NULL);

                    this->host = (T *) this->dev->getHost();

                } else {

                    this->dev = NULL;
                    if (host_mem == NULL) {
                        this->sp_host = std::shared_ptr<T>(new T[this->n_cells]);
                        this->host = this->sp_host.get();
                    } else {
                        this->host = host_mem;
                    }

                }
            } else {
                this->dims.assign(this->dims.size(), 0);
                this->dev = NULL;
                this->host = NULL;
            }

            if (this->host != NULL) {
                memset((void*) this->host, 0, sizeof (T) * this->n_cells);
            }

            this->hist_count = NULL;
            this->mean = 0;
            this->min = std::numeric_limits<T>::min();
            this->max = std::numeric_limits<T>::max();
            this->stdev = 0;
        }

        void init(std::vector<T>& data, std::vector<size_t> dims,
                T* host_mem = NULL) {
            this->init(dims, host_mem);
            memmove(this->getHost(), &data[0], data.size() * sizeof (T));
        }

        bool operator==(OCLMatrix<T>& right) {
            return this->dev == right.dev && this->host == right.host;
        }

        ~OCLMatrix() {
        }

        static std::shared_ptr<OCLMatrix<T> > hstack(
                std::vector<std::shared_ptr<OCLMatrix<T> > >& others) {
            auto context = others.at(0)->context;
            auto kernels = others.at(0)->kernels;

            size_t dev_fresh_count = 0;
            size_t host_fresh_count = 0;

            std::shared_ptr<OCLMatrix<T> > new_matrix(
                    new OCLMatrix<T>(context, kernels));
            std::vector<size_t> new_dims(others.at(0)->dims.begin(),
                    others.at(0)->dims.end());

            new_dims[1] = 0; // Columns
            for (std::shared_ptr<OCLMatrix<T> > x : others) {
                new_dims[0] = std::min(new_dims[0], x->dims[0]); // Min rows
                new_dims[1] += x->dims[1]; // Sum columns
                if (x->host_is_fresh())
                    host_fresh_count += x->size;
                else
                    dev_fresh_count += x->size;
            }

            new_matrix->init(new_dims);

            size_t col_index = 0;
            for (std::shared_ptr<OCLMatrix<T> >& o : others) {
                new_matrix->read_rect(*o, 0, 0, 0, col_index, o->rows(), o->cols());
                col_index += o->cols();
            }

            return new_matrix;
        }

        static std::shared_ptr<OCLMatrix<T> > vstack(
                std::vector<std::shared_ptr<OCLMatrix<T> > >& others,
                std::vector<std::vector<size_t>*>& row_maps) {
            std::shared_ptr<OCLMatrix<T> > other;
            OpenCL::spContext context(NULL);
            OpenCL::KernelMap* kernels = NULL;
            size_t width = 0;
            size_t total_height = 0;
            for (std::shared_ptr<OCLMatrix<T> > o : others) {

                total_height += o->rows();
                width = o->cols();
                other = o;
            }

            std::shared_ptr<OCLMatrix<T> > mx(
                    new OCLMatrix<T>(other->context, other->kernels));
            mx->init({total_height, width});

            if (row_maps.size() == others.size()) {
                for (size_t o_idx = 0; o_idx < others.size(); o_idx++) {
                    std::shared_ptr<OCLMatrix<T> > o = others.at(o_idx);
                    std::vector<size_t>& omp = *row_maps.at(o_idx);
                    size_t row_size = sizeof (T) * width;
                    // We must have a complete map from o -> mx
                    assert(omp.size() == o->rows());
                    for (size_t r = 0; r < o->rows(); r++) {
                        memmove(mx->row_ref(omp.at(r)), o->row_ref(r), row_size);
                    }
                }
            } else {
                size_t row_index = 0;
                for (std::shared_ptr<OCLMatrix<T> > o : others) {
                    mx->read_rect(*o, 0, 0, row_index, 0, o->rows(), o->cols());
                    row_index += o->rows();
                }
            }

            return mx;
        }

        static std::pair<std::shared_ptr<OCLMatrix<T> >,
        std::shared_ptr<OCLMatrix<T> > > partition(
                std::vector<T> data, size_t width, std::vector<size_t>& rows_in_a,
                std::vector<size_t>& rows_in_b, const OpenCL::spContext context,
                const OpenCL::KernelMap * kernels) {
            size_t a_i = 0;
            size_t b_i = 0;
            std::shared_ptr<OCLMatrix<T> > A(new OCLMatrix<T>(context, kernels));
            std::shared_ptr<OCLMatrix<T> > B(new OCLMatrix<T>(context, kernels));
            size_t copy_size = sizeof (T) * width;

            A->init({rows_in_a.size(), width});
            B->init({rows_in_b.size(), width});

            for (size_t r : rows_in_a)
                memmove(A->row_ref(a_i++), &data[r * width], copy_size);

            for (size_t r : rows_in_b)
                memmove(B->row_ref(b_i++), &data[r * width], copy_size);

            return std::make_pair(A, B);
        }

        static std::pair<std::shared_ptr<OCLMatrix<T> >,
        std::shared_ptr<OCLMatrix<T> > > random_partition(
                std::vector<T> data, size_t width, decimal_t p,
                std::vector<size_t>& rows_in_data, std::vector<size_t>& rows_in_holdout,
                const OpenCL::spContext context, const OpenCL::KernelMap * kernels) {
            std::shared_ptr<OCLMatrix<T> > A(new OCLMatrix<T>(context, kernels));
            std::shared_ptr<OCLMatrix<T> > B(new OCLMatrix<T>(context, kernels));
            size_t rows = data.size() / width;
            size_t ai = 0;
            size_t bi = 0;
            size_t b_rows = rows * p;
            size_t a_rows = rows - b_rows;
            size_t copy_size = width * sizeof (T);
            float rnum;
            assert((data.size() % width) == 0);

            A->init({a_rows, width});
            B->init({b_rows, width});

            A->getHost();
            B->getHost();

            for (size_t i = 0; i < rows; i++) {
                rnum = (float) rand() / (float) RAND_MAX;
                if (((bi < b_rows) && rnum < p) || ai >= a_rows) {
                    rows_in_holdout.push_back(i);
                    memmove(B->row_ref(bi++), &data[i * width], copy_size);
                } else {
                    rows_in_data.push_back(i);
                    memmove(A->row_ref(ai++), &data[i * width], copy_size);
                }
            }

            return std::make_pair(A, B);
        }

        void churn(OCLMatrix<T> & other, float pswap = 0.2) {
            assert(this->cols() == other.cols());
            assert(this->rows() == other.rows());

            float rnum;

            for (unsigned int i = 0; i < this->rows(); i++) {
                rnum = (float) rand() / (float) RAND_MAX;
                if (rnum > pswap) {
                    this->swap_row(other, i, i);
                }
            }
        }

        size_t sample_rows(OCLMatrix<T>& source) {
            if (source.cols() != this->cols())
                return 0;

            if (source.n_cells == this->n_cells)
                return this->read_from(source, 0, 0, this->n_cells);

            auto src_rows = source.rows();
            auto read_width = this->cols();
            size_t total_read = 0;

            for (size_t i = 0; i < this->rows(); i++) {
                this->read_from(source, i * read_width, (rand() % src_rows) * read_width,
                        read_width);
            }

            return total_read;
        }

        T * getHost() {
            if (this->dev != NULL)
                return (T*) this->dev->getHost();
            else
                return this->host;
        }

        const T * getHostConst() {
            if (this->dev != NULL)
                return (const T*) this->dev->getHostConst();
            else
                return (const T*) this->host;
        }

        cl_mem getCL() {
            if (this->dev != NULL) {
                return this->dev->getCL();
            } else {
                return NULL;
            }
        }

        const cl_mem getCLConst() {
            if (this->dev != NULL)
                return this->dev->getCLConst();
            else
                return NULL;
        }

        void read_from_dev() {
            this->dev->read();
        }

        void write_to_dev() {
            this->dev->write();
        }

        void append_to(std::vector<T>& vec) {
            auto h = this->getHost();
            vec.reserve(vec.size() + this->n_cells);
            for (size_t x = 0; x < this->n_cells; x++)
                vec.push_back(h[x]);
        }

        void set_dims(std::vector<size_t> new_dims) {
#ifndef NDEBUG
            for (size_t x = 0; x < new_dims.size(); x++)
                assert(this->_init_dims[x] >= new_dims[x]);
#endif
            this->dims = new_dims;
        }

        void reset_dims() {
            this->dims = this->_init_dims;
        }

        bool same_dims(OCLMatrix<T>& other) {
            if (this->dims.size() == other.dims.size())
                return std::equal(this->dims.begin(), this->dims.end(),
                    other.dims.begin());
            else
                return false;
        }

        void copy_to(OCLMatrix<T>& dest) {
            if (!(*this == dest)) {

                if (this->use_cl(dest)) {
                    clEnqueueCopyBuffer(this->dev->queue, this->getCLConst(), dest.getCL(),
                            0, 0, this->size, 0, NULL, NULL);
                } else {
                    assert(dest.n_cells == this->n_cells);
                    memmove(dest.getHost(), this->getHostConst(), this->size);
                }
            }
        }

        size_t read_from(OCLMatrix<T>& src, size_t dest_offset, size_t src_offset,
                size_t cell_count) {
            assert(src.check());

            if (*this == src)
                return 0;

            cell_count = std::min(cell_count, src.n_cells - src_offset);
            cell_count = std::min(cell_count, this->n_cells - dest_offset);

            size_t copy_size = cell_count * sizeof (T);

            if (this->use_cl(src)) {
                const cl_mem src_mem = src.getCLConst();
                clEnqueueCopyBuffer(this->dev->queue, src_mem, this->getCL(),
                        src_offset * sizeof (T), dest_offset * sizeof (T),
                        copy_size, 0, NULL, NULL);
            } else {
                // Doing this avoids a possible extraneous read
                if (this->dev != NULL)
                    this->dev->touched_by_device = false;

                this->getHost();
                src.getHostConst();

                memmove(this->ref(dest_offset), src.ref(src_offset), copy_size);
            }

            return cell_count;
        }

        size_t read_from(std::vector<T>& src, size_t offset = 0) {
            assert(this->n_cells >= (src.size() + offset));
            this->getHost();

            for (size_t i = 0; i < src.size(); i++) {
                this->at(offset + i) = src.at(i);
            }
            return offset + src.size();
        }

        void read_row(OCLMatrix<T>& M, size_t src_row, size_t dest_row) {
            assert(this->cols() == M.cols());
            this->getHost();
            M.getHostConst();
            for (size_t c = 0; c < this->cols(); c++)
                this->at(dest_row, c) = M.at(src_row, c);
        }

        void swap_row(OCLMatrix<T>& dest, size_t self_row, size_t dest_row) {
            assert(this->cols() == dest.cols());
            T * selfHost = this->getHost() + (self_row * this->cols());
            T * destHost = dest.getHost() + (dest_row * dest.cols());
            T t[this->cols()];
            size_t moveSize = this->cols() * sizeof (T);

            memmove(&t, selfHost, moveSize);
            memmove(selfHost, destHost, moveSize);
            memmove(destHost, &t, moveSize);
        }

        void read_col(OCLMatrix<T>& M, size_t src_col, size_t dest_col) {
            assert(this->rows() == M.rows());
            this->getHost();
            M.getHostConst();
            for (size_t r = 0; r < this->rows(); r++)
                this->at(r, dest_col) = M.at(r, src_col);
        }

        void read_row_as_col(OCLMatrix<T>& M, size_t src_row, size_t dest_col) {
            assert(this->rows() == M.cols());
            this->getHost();
            M.getHostConst();
            for (size_t r = 0; r < this->rows(); r++)
                this->at(r, dest_col) = M.at(src_row, r);
        }

        void read_col_as_row(OCLMatrix<T>& M, size_t src_col, size_t dest_row) {
            assert(this->cols() == M.rows());
            this->getHost();
            M.getHostConst();
            for (size_t c = 0; c < this->cols(); c++)
                this->at(dest_row, c) = M.at(c, src_col);
        }

        void read_rect(OCLMatrix<T>& src, size_t src_row = 0, size_t src_col = 0,
                size_t dest_row = 0, size_t dest_col = 0, size_t rows = 0,
                size_t cols = 0) {
            rows = std::min(rows, src.rows() - src_row);
            rows = std::min(rows, this->rows() - dest_row);

            cols = std::min(cols, src.cols() - src_col);
            cols = std::min(cols, this->cols() - dest_col);

            if (rows == 0)
                rows = src.rows();
            if (cols == 0)
                cols = src.cols();

            if (this->use_cl(src)) {
                // Target transfers to host
                size_t src_origin[] = {src_col, src_row, 0};
                size_t dest_origin[] = {dest_col, dest_row, 0};
                size_t region[] = {cols * sizeof (T), rows, 1};

                clEnqueueCopyBufferRect(this->dev->queue, src.getCLConst(), this->getCL(),
                        src_origin, dest_origin, region,
                        src.cols() * sizeof (T), 0,
                        this->cols() * sizeof (T), 0, 0, NULL, NULL);

            } else {

                this->getHost();
                src.getHost();
                // Target transfers to host
                size_t copy_size = cols * sizeof (T);
                for (size_t r = 0; r < rows; r++)
                    memmove(this->row_ref(r + dest_row) + dest_col,
                        src.row_ref(r + src_row) + src_col, copy_size);

            }
        }

        void read_from(std::istream& f) {
            // Doing this avoids a possible extraneous read
            if (this->dev != NULL)
                this->dev->touched_by_device = false;
            T* dest = this->getHost();
            f.read((char*) dest, sizeof (T) * this->n_cells);
        }

        void write_to(std::ostream& f) {
            this->getHostConst();
            size_t i = 0;
            while (i < this->n_cells) {
                f.write((const char*) this->ref(i++), sizeof (T));
            }
        }

        void dump_stats(int nbins) {
            this->make_stats(nbins);
            this->print_stats(std::cout);
        }

        void statsToJson(std::stringstream& str, size_t nbins = 100) {
            this->make_stats(nbins);
            str << "{";
            str << "\"n\":" << this->n_cells;
            str << ",\"min\":" << this->min;
            str << ",\"max\":" << this->max;
            str << ",\"mean\":" << this->mean;
            str << ",\"stdev\":" << this->stdev;
            str << ",\"histogram\": [";

            if (nbins > 0) {
                str << this->hist_count[0];
            }

            for (int i = 1; i < this->n_hist_bins; i++) {
                str << "," << this->hist_count[i];
            }
            str << "] }";
        }

        void make_stats(int n_hist_bins = 100) {
            this->getHostConst();
            int i;

            if (this->hist_count == NULL || this->n_hist_bins != n_hist_bins) {
                this->sp_hist_count = std::shared_ptr<size_t>(new size_t[n_hist_bins]);
                this->hist_count = sp_hist_count.get();
            }

            for (i = 0; i < n_hist_bins; i++) {
                this->hist_count[i] = 0;
            }

            this->n_hist_bins = n_hist_bins;

            // Stats are on the CPU
            T v;
            const T * data = this->getHostConst();

            T v_min = std::numeric_limits<T>::max();
            T v_max = std::numeric_limits<T>::min();

            T sum = 0;
            T mean = 0;

            for (i = 0; i < this->n_cells; i++) {
                v = data[i];

                assert(std::isfinite(v));

                v_min = std::min(v_min, v);
                v_max = std::max(v_max, v);
                sum += v;
            }

            this->min = v_min;
            this->max = v_max;

            mean = sum / this->n_cells;

            T stdev_sum = 0;
            T span = ((v_max - v_min) == 0) ? 1 : v_max - v_min;
            T binstep = span / n_hist_bins;
            double inv_step = 1.0 / binstep;
            int bin;

            for (i = 0; i < this->n_cells; i++) {
                v = data[i];
                stdev_sum += pow(v - mean, 2);

                bin = std::min(((int) floor((v - this->min) * inv_step)),
                        n_hist_bins - 1);
                assert((bin >= 0) && (bin < n_hist_bins));
                this->hist_count[bin]++;
            }

            this->stdev = sqrt(stdev_sum / this->n_cells);
            this->mean = mean;

        }

        std::shared_ptr<OCLMatrix<T> > covar_rows() {

            std::vector<double> exp(this->rows());
            std::vector<double> xv(this->cols());
            std::shared_ptr<OCLMatrix<T> > cv(this->context, this->kernels);
            cv->init({this->rows(), this->rows()});

            // Get mean
            for (size_t r = 0; r < this->rows(); r++) {
                double sum = 0;
                for (size_t c = 0; c < this->cols(); c++) {
                    sum += this->at(r, c);
                }
                exp[r] = sum / this->cols();
            }

            for (size_t x = 0; x < this->rows(); x++) {

                for (size_t c = 0; c < this->cols(); c++) {
                    xv[c] = this->at(x, c) - exp[x];
                }

                for (size_t y = 0; y < this->rows(); y++) {
                    double sum = 0.0;

                    for (size_t c = 0; c < this->cols(); c++)
                        sum += xv[c] * (this->at(y, c) - exp[y]);
                    cv->at(x, y) = sum / this->cols();
                }

            }

            return cv;
        }

        std::shared_ptr<OCLMatrix<T> > covar_columns() {
            std::vector<double> exp(this->cols());
            std::vector<double> xv(this->rows());
            std::shared_ptr<OCLMatrix<T> > cv(this->context, this->kernels);
            cv->init({this->cols(), this->cols()});

            // Get mean
            for (size_t c = 0; c < this->cols(); c++) {
                double sum = 0;
                for (size_t r = 0; r < this->rows(); r++)
                    sum += this->at(r, c);
                exp[c] = sum / this->rows();
            }

            for (size_t x = 0; x < this->cols(); x++) {
                for (size_t r = 0; r < this->rows(); r++)
                    xv[r] = this->at(r, x) - exp[x];

                for (size_t y = 0; y < this->cols(); y++) {
                    double sum = 0.0;
                    for (size_t r = 0; r < this->rows(); r++)
                        sum += xv[r] * (this->at(r, y) - exp[y]);
                    cv->at(x, y) = sum / this->rows();
                }
            }

            return cv;
        }

        /**
         * @brief orthonormalize This will orthonormalize a matrix using the Modified Gram-Schmidt process.
         *
         */
        void orthonormalize() {
            for (size_t i = 0; i < this->cols(); i++) {
                this->orthonormalize(i);
            }
        }

        void orthonormalize(size_t col) {
            for (size_t j = 0; j < col; j++) {
                T dotp = this->dot_col_col(*this, col, j);
                for (size_t i = 0; i < this->rows(); i++)
                    this->at(i, col) -= dotp * this->at(i, j);
            }
        }

        /**
         * @brief PCA Fast PCA based on:
         *              Sharma, Alok, and Kuldip K. Paliwal. "Fast principal component analysis using fixed-point algorithm." Pattern Recognition Letters 28.10 (2007): 1151-1155.
         * @param n Number of components to extract
         * @return
         */
        std::shared_ptr<OCLMatrix<T> > PCA(size_t h, double error = 0.01) {
            size_t x = 0;
            OCLMatrix<T> v_p(NULL, NULL);
            OCLMatrix<T> v_p_prev(NULL, NULL);
            v_p.init({this->cols(), 1});
            v_p_prev.init({this->cols(), 1});

            std::shared_ptr<OCLMatrix<T> > covar = this->covar_columns();
            std::shared_ptr<OCLMatrix<T> > comps(
                    new OCLMatrix<T>(this->context, this->kernels));
            comps->init({this->rows(), h});
            comps->set(1.0);

            for (size_t p = 0; p < h; p++) {
                T e = std::numeric_limits<T>::infinity();
                x = 0;
                v_p.read_col(*comps, p, 0);
                v_p_prev.read_col(*comps, p, 0);

                // The paper says that the number of iterations is small (2-5), but
                // we put a limit on it anyway
                while ((x++ < 50) && (e > error)) {
                    covar->cross(v_p, v_p);
                    comps->read_col(v_p, 0, p);
                    comps->orthonormalize(p);
                    comps->norm_col(p);
                    v_p.read_col(*comps, p, 0);
                    e = v_p.dot_col_col(v_p_prev, 0, 0) - 1;
                }
            }

            return comps;
        }

        void norm() {
            double sum = 0.0;
            for (size_t i = 0; i < this->n_cells; i++)
                sum += ::pow(this->at(i), 2);
            sum = ::sqrt(sum);
            for (size_t i = 0; i < this->n_cells; i++)
                this->at(i) /= sum;
        }

        void norm_row(size_t row) {
            T s = 0;
            for (size_t i = 0; i < this->cols(); i++)
                s += this->at(row, i);
            for (size_t i = 0; i < this->cols(); i++)
                this->at(row, i) /= s;
        }

        void norm_rows() {
            for (size_t r = 0; r < this->rows(); r++)
                this->norm_row(r);
        }

        void norm_col(size_t col) {
            T s = 0;
            for (size_t i = 0; i < this->rows(); i++)
                s += this->at(i, col);
            for (size_t i = 0; i < this->rows(); i++)
                this->at(i, col) /= s;
        }

        void norm_cols() {
            for (size_t c = 0; c < this->cols(); c++)
                this->norm_col(c);
        }

        std::shared_ptr<OCLMatrix<T> > cross(OCLMatrix<T>& B) {
            assert(this->cols() == B.rows());
            std::shared_ptr<OCLMatrix<T> > cp(
                    new OCLMatrix<T>(this->context, this->kernels));
            cp->init({this->rows(), B.cols()});
            this->cross(B, *cp);
            return cp;
        }

        void cross(OCLMatrix<T>& B, OCLMatrix<T>& dest) {
            assert(B.cols() == dest.rows());
            assert(this->cols() == B.rows());

            for (size_t i = 0; i < this->rows(); i++) {
                for (size_t j = 0; j < B.cols(); j++) {
                    double ij_dot = 0.0;
                    for (size_t x = 0; x < this->cols(); x++)
                        ij_dot += this->at(i, x) * B.at(x, j);
                    dest.at(i, j) = ij_dot;
                }
            }
        }

        T dot_row_row(OCLMatrix<T>& M, size_t my_row, size_t m_row) {
            assert(this->cols() == M.cols());
            T dot_prod = 0.0;
            for (size_t c = 0; c < this->cols(); c++)
                dot_prod += this->at(my_row, c) * M.at(m_row, c);
            return dot_prod;
        }

        T dot_row_col(OCLMatrix<T>& M, size_t my_row, size_t m_col) {
            assert(this->cols() == M.rows());
            T dot_prod = 0.0;
            for (size_t x = 0; x < this->cols(); x++)
                dot_prod += this->at(my_row, x) * M.at(x, m_col);
            return dot_prod;
        }

        T dot_col_row(OCLMatrix<T>& M, size_t my_col, size_t m_row) {
            assert(this->rows() == M.cols());
            T dot_prod = 0.0;
            for (size_t x = 0; x < this->rows(); x++)
                dot_prod += this->at(x, my_col) * M.at(m_row, x);
            return dot_prod;
        }

        T dot_col_col(OCLMatrix<T>& M, size_t my_col, size_t m_col) {
            assert(this->rows() == M.rows());
            T dot_prod = 0.0;
            for (size_t x = 0; x < this->rows(); x++)
                dot_prod += this->at(x, my_col) * M.at(x, m_col);
            return dot_prod;
        }

        size_t getReadCount() {
            if (this->dev != NULL) {
                return this->dev->getReadCount();
            } else {
                return 0;
            }
        }

        size_t getWriteCount() {
            if (this->dev != NULL) {
                return this->dev->getWriteCount();
            } else {
                return 0;
            }
        }

        double mae(OCLMatrix<T>& other) {

            auto A = this->getHostConst();
            auto B = other.getHostConst();
            double error = 0.0;
            assert(this->n_cells == other.n_cells);
            for (size_t i = 0; i < this->n_cells; i++) {
                error += fabs(A[i] - B[i]);
            }
            return error / this->n_cells;

        }

        double nrmse(OCLMatrix<T>& other) {
            this->getHostConst();
            other.getHostConst();

            double rmse = 0.0;
            T vmin = std::numeric_limits<T>::max();
            T vmax = std::numeric_limits<T>::min();

            if (other.n_cells != this->n_cells)
                return std::numeric_limits<double>::infinity();

            for (size_t i = 0; i < this->n_cells; i++) {
                rmse += ::pow(this->at(i) - other.at(i), 2);
                vmin = std::min(vmin, std::min(this->at(i), other.at(i)));
                vmax = std::max(vmax, std::max(this->at(i), other.at(i)));
            }

            rmse = ::sqrt(rmse / this->n_cells);
            assert(std::isfinite(rmse));

            return rmse / (vmax - vmin);
        }

        void sync_host() {
            this->getHostConst();
        }

        void transpose_to(OCLMatrix<T>& right) {
            if (this->rows() == right.cols() && this->cols() == right.rows()) {
                size_t r, c;
                size_t r_end = this->rows();
                size_t c_end = this->cols();
                for (r = 0; r < r_end; r++) {
                    for (c = 0; c < c_end; c++)
                        right.at(c, r) = this->at(r, c);
                }
            }
        }

        inline T& at(size_t index) {
            assert(this->n_cells > index);
            return this->host[index];
        }

        inline T& at(size_t row, size_t column) {
            assert(this->dims.size() == 2);
            assert(this->dims.at(0) > row);
            assert(this->dims.at(1) > column);
            return this->host[this->idx(row, column)];
        }

        T& at(std::vector<size_t> indices) {
            assert(this->dims.size() >= indices.size());
            size_t hi = indices.size() - 1;
            size_t idx = this->_ndim_idx(indices, indices.size() - 1);
            assert(idx < this->n_cells);
            return this->host.at(idx);
        }

        inline T* ref(size_t idx) {
            assert(idx < this->n_cells);
            return this->host + idx;
        }

        inline T* ref(size_t r, size_t c) {
            size_t i = this->idx(r, c);
            assert(i < this->n_cells);
            return this->host + i;
        }

        void set(T v) {

#ifdef clEnqueueFillBuffer
            if (this->dev == NULL) {
#endif

                T* h = this->getHost();
                for (size_t i = 0; i < this->n_cells; i++) {
                    h[i] = v;
                }

#ifdef clEnqueueFillBuffer
            } else {

                clEnqueueFillBuffer(this->dev->queue,
                        this->getCL(),
                        &v,
                        sizeof (v),
                        0,
                        this->dev->size,
                        0,
                        NULL,
                        NULL);
            }
#endif

        }

        void fillRandNormal(decimal_t mean = 0.0, decimal_t stdev = 1.0) {

            std::normal_distribution<decimal_t> d(mean, stdev);

            for (size_t i = 1; i < this->n_cells; i++) {
                this->at(i) = 0;
            }

            for (size_t i = 1; i < this->n_cells; i++) {
                this->at(i) = d(ailab::gen);
            }
        }

        void fillRandUniform(decimal_t low = 0.0, decimal_t high = 1.0) {

            std::uniform_real_distribution<decimal_t> d(low, high);
            decimal_t * hm = this->getHost();
            for (size_t i = 1; i < this->n_cells; i++) {
                hm[i] = d(ailab::gen);
            }

        }

        T* row_ref(size_t num) {
            return this->host + this->idx(num, 0);
        }

        size_t _ndim_idx(std::vector<size_t>& indices, size_t i) {
            if (i == 0)
                return this->dims[i + 1] * indices[i];
            else
                return indices[i] + (this->dims[i] * this->_ndim_idx(indices, i - 1));
        }

        void fill_with(std::iterator<std::forward_iterator_tag, T> begin,
                std::iterator<std::forward_iterator_tag, T> end,
                int index = 0) {
            T* v = this->getHost();
            while (begin != end && index < this->n_cells)
                v[index++] = *begin++;
        }

        size_t rows() {
            return (this->dims.size() >= 1) ? this->dims.at(0) : 0;
        }

        size_t cols() {
            return (this->dims.size() >= 2) ? this->dims.at(1) : 0;
        }

        size_t depth() {
            return (this->dims.size() >= 3) ? this->dims.at(2) : 0;
        }

        /**
         * @brief write_as_img Writes this matrix out as a grayscale png image
         *          All values are scaled to fit bewteen 0 and 255.
         * @param path
         * @return
         */
        bool write_as_img(std::string path, T vmin = std::numeric_limits<T>::max(),
                T vmax = std::numeric_limits<T>::min()) {
            if (this->dims.size() != 2) {
                std::cerr << "Sorry, I can only write images from 2D matrics"
                        << std::endl;
                return false;
            }

            if (!endswith(path, ".pgm"))
                path.append(".pgm");

            std::ofstream f(path);

            if (f.good()) {
                size_t v;
                const T* m = this->getHostConst();
                for (size_t x = 0; x < this->n_cells; x++) {
                    vmin = std::min(m[x], vmin);
                    vmax = std::max(m[x], vmax);
                }

                size_t range = vmax - vmin;

                f << "P2" << std::endl << this->cols() << " " << this->rows() << "\n"
                        << range << std::endl;
                for (size_t r = 0; r < this->rows(); r++) {
                    v = (this->at(r, 0) - vmin);
                    f << v;
                    for (size_t c = 1; c < this->cols(); c++) {
                        v = (this->at(r, c) - vmin);
                        f << " " << v;
                    }
                    f << std::endl;
                }

                f.close();
                return true;
            } else {
                std::cerr << "Could not open " << path << std::endl;
            }
            return false;
        }

        void write_to_ascii(std::ostream stream, bool row_major = true, char sep =
                '\t') {
            size_t cols = 1;
            if (this->dims.size() > 1)
                cols = this->dims.at(1);

            this->getHostConst();

            if (row_major) {
                for (size_t r = 0; r < this->rows(); r++) {
                    stream << this->at(r, 0);
                    for (size_t c = 1; c < cols; c++)
                        stream << sep << this->at(r, c);
                    stream << std::endl;
                }
            } else {
                for (size_t c = 0; c < cols; c++) {
                    stream << this->at(0, c);
                    for (size_t r = 1; r < this->rows(); r++)
                        stream << sep << this->at(r, c);
                    stream << std::endl;
                }
            }
        }

        bool write_to_csv(std::string path, bool row_major = true, char sep = ',') {
            std::ofstream f(path);

            if (f.good()) {
                this->write_to_ascii(f, row_major, sep);
                f.close();
                return true;
            } else {
                std::cerr << "Could not open " << path << std::endl;
            }
            return false;
        }

        bool device_is_fresh() {
            if (this->dev == NULL)
                return false;
            else
                return this->dev->touched_by_device;
        }

        bool host_is_fresh() {
            return (this->dev == NULL) || this->dev->touched_by_host;
        }

        bool check(bool initial = false) {

            if (this->n_cells == 0) {
                return false;
            }

            if (initial) {
                this->getHost();

                if (this->dev != NULL) {
                    this->dev->write();
                }
            }

            if (this->dev != NULL) {
                this->dev->load();
                if (this->dev->size != this->size)
                    return false;

                if (!this->dev->is_readable_by_kernel && !this->dev->is_writable_by_kernel) {
                    return false;
                }
            }

            const T * pv = this->getHostConst();
            for (size_t i = 0; i < this->n_cells; i++) {
                if (pv[i] != pv[i]
                        || pv[i] == std::numeric_limits<T>::infinity()
                        || pv[i] == -std::numeric_limits<T>::infinity()) {
                    return false;
                }
            }

            return true;
        }

        size_t n_cells;
        size_t size;
        std::vector<size_t> dims;
        std::vector<std::string> column_labels;

        int n_hist_bins;
        size_t * hist_count;

        double mean;
        double stdev;
        T min;
        T max;

    protected:

        T * host;
        OpenCL::spMemory dev;
        OpenCL::spMemory errMeasure;
        std::vector<size_t> _init_dims;

        std::shared_ptr<T> sp_host;
        std::shared_ptr<size_t> sp_hist_count;

        bool use_cl(OCLMatrix<T>& other) {
            if (this->dev != NULL) {
                size_t size_on_device = 0;
                size_t size_on_host = 0;

                if (this->device_is_fresh())
                    size_on_device += this->n_cells;
                else
                    size_on_host += this->n_cells;

                if (other.device_is_fresh())
                    size_on_device += other.n_cells;
                else
                    size_on_host += other.n_cells;

                return size_on_device > size_on_host;
            }
            return false;
        }

        inline size_t idx(size_t row, size_t column) {
#ifdef COLUMN_MAJOR_MATRICES
            return row + this->dims[0] * column;
#else
            return row * this->dims[1] + column;
#endif
        }

    };

}
#endif // OCLMATRIX_H
